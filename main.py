import jiwer
import pandas as pd
import re
import time
import soundfile as sf
import os
import numpy as np
from scipy import signal
import tritonclient.grpc.aio as grpcclient_aio  # ← async gRPC client
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype
import json
import asyncio
import warnings
from itertools import groupby
from dataclasses import dataclass
from typing import List, Optional, Tuple
import webrtcvad
import collections

try:
    from pyctcdecode import BeamSearchDecoderCTC
    from pyctcdecode.decoder import build_ctcdecoder  # ← Правильный импорт
    PYCTCDECODE_AVAILABLE = True
    print("pyctcdecode is available")
except ImportError:
    PYCTCDECODE_AVAILABLE = False
    print("pyctcdecode not available, using greedy decoder only")

# Константы из T-One
LABELS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "  # Словарь из decoder.py
BLANK_TOKEN_ID = len(LABELS)  # blank token - последний в словаре
CHUNK_SIZE = 2400  # Из StreamingCTCModel.AUDIO_CHUNK_SAMPLES
STATE_SIZE = 219729  # Из StreamingCTCModel.STATE_SIZE
SAMPLE_RATE = 8000  # Из StreamingCTCModel.SAMPLE_RATE


@dataclass
class CharTimestamp:
    char: str
    start_ms: float
    end_ms: float

@dataclass
class PhraseTimestamp:
    text: str
    start_ms: float
    end_ms: float
    
@dataclass
class LogprobPhrase:
    """Класс для фразы из logprob splitter"""
    logprobs: np.ndarray
    start_frame: int
    end_frame: int

@dataclass 
class StreamingLogprobSplitterState:
    """Состояние для StreamingLogprobSplitter"""
    past_logprobs: np.ndarray
    offset: int = 0

@dataclass
class AudioSegment:
    """Сегмент аудио с временными границами"""
    audio_data: np.ndarray
    start_ms: float
    end_ms: float

class StreamingLogprobSplitter:
    """Упрощенная версия логики разбиения логпробабилити на фразы из T-One"""
    
    SILENCE_THRESHOLD = 0.8  # вероятность (0 - 1)
    MIN_SILENCE_DURATION = 20  # в акустических фреймах
    SPEECH_EXPAND_SIZE = 3  # в акустических фреймах
    MAX_PHRASE_DURATION = 2000  # в акустических фреймах
    
    def __init__(self):
        pass
    
    def forward(self, logprobs: np.ndarray, state: Optional[StreamingLogprobSplitterState] = None, 
                is_last: bool = False) -> Tuple[List[LogprobPhrase], StreamingLogprobSplitterState]:
        """Обрабатывает чанк логпробабилити и извлекает завершенные сегменты"""
        
        if state is None:
            state = StreamingLogprobSplitterState(
                past_logprobs=np.zeros((0, 35), dtype=np.float32),
                offset=0
            )
        
        # Объединяем старые логпробы с новыми
        logprobs = np.concatenate((state.past_logprobs, logprobs), axis=0)
        
        # Если вероятность пробела + blank токенов меньше порога, считаем это речью
        # Последние два токена в словаре - пробел и blank
        is_speech = np.exp(logprobs[..., -2:]).sum(axis=-1) <= self.SILENCE_THRESHOLD
        
        phrases = []
        last_phrase = 0
        
        # Простая логика: если это последний чанк, создаем фразу из всех оставшихся логпробов
        if is_last and len(logprobs) > 0:
            phrase = LogprobPhrase(
                logprobs=logprobs,
                start_frame=state.offset,
                end_frame=state.offset + len(logprobs)
            )
            phrases.append(phrase)
            last_phrase = len(logprobs)
        else:
            # Для промежуточных чанков можем реализовать более сложную логику
            # Пока что просто ждем последний чанк
            last_phrase = 0
        
        # Обновляем состояние
        next_offset = state.offset + last_phrase
        new_state = StreamingLogprobSplitterState(
            past_logprobs=logprobs[last_phrase:],
            offset=next_offset
        )
        
        return phrases, new_state

class CTCAlignment:
    """Класс для получения alignment между текстом и CTC логпробабилити"""
    
    def __init__(self, labels: str = LABELS):
        self.labels = labels
        self.char_to_id = {char: i for i, char in enumerate(labels)}
        self.blank_id = len(labels)
    
    def text_to_ids(self, text: str) -> List[int]:
        """Конвертирует текст в последовательность ID"""
        return [self.char_to_id[char] for char in text if char in self.char_to_id]
    
    def force_align(self, logprobs: np.ndarray, text: str, frame_duration_ms: float = 30.0) -> List[CharTimestamp]:
        """
        Выполняет принудительное выравнивание между текстом и логпробабилити
        Использует modified Viterbi algorithm для CTC alignment
        """
        if not text.strip():
            return []
        
        text_ids = self.text_to_ids(text)
        if not text_ids:
            return []
        
        T, V = logprobs.shape  # time, vocab
        N = len(text_ids)      # text length
        
        # Создаем расширенную последовательность с blank токенами
        # [blank, char1, blank, char2, blank, ...]
        extended_seq = [self.blank_id]
        for char_id in text_ids:
            extended_seq.extend([char_id, self.blank_id])
        
        S = len(extended_seq)
        
        # Dynamic programming таблица
        dp = np.full((T, S), -np.inf)
        
        # Инициализация
        dp[0, 0] = logprobs[0, self.blank_id]  # blank
        if S > 1:
            dp[0, 1] = logprobs[0, extended_seq[1]]  # first char
        
        # Forward pass
        for t in range(1, T):
            for s in range(S):
                label = extended_seq[s]
                
                # Оставаться в том же состоянии
                dp[t, s] = max(dp[t, s], dp[t-1, s] + logprobs[t, label])
                
                # Переход из предыдущего состояния
                if s > 0:
                    dp[t, s] = max(dp[t, s], dp[t-1, s-1] + logprobs[t, label])
                
                # Пропуск повторяющегося символа (только для не-blank)
                if s > 1 and extended_seq[s] != self.blank_id and extended_seq[s-2] == extended_seq[s]:
                    dp[t, s] = max(dp[t, s], dp[t-1, s-2] + logprobs[t, label])
        
        # Backtrack для получения лучшего пути
        path = []
        s = S - 1
        for t in range(T-1, -1, -1):
            path.append((t, s, extended_seq[s]))
            
            if t > 0:
                # Найти лучший предыдущий переход
                candidates = []
                
                # Остаться в том же состоянии
                candidates.append((s, dp[t-1, s]))
                
                # Переход из предыдущего состояния
                if s > 0:
                    candidates.append((s-1, dp[t-1, s-1]))
                
                # Пропуск повторяющегося символа
                if s > 1 and extended_seq[s] != self.blank_id and extended_seq[s-2] == extended_seq[s]:
                    candidates.append((s-2, dp[t-1, s-2]))
                
                # Выбрать лучший переход
                s = max(candidates, key=lambda x: x[1])[0]
        
        path.reverse()
        
        # Извлекаем временные метки для символов
        char_timestamps = []
        current_char = None
        char_start = None
        
        for t, s, label_id in path:
            if label_id != self.blank_id:  # не blank
                char = self.labels[label_id]
                
                if current_char != char or char_start is None:
                    # Закрываем предыдущий символ
                    if current_char is not None and char_start is not None:
                        char_timestamps.append(CharTimestamp(
                            char=current_char,
                            start_ms=round(char_start * frame_duration_ms, 2),
                            end_ms=round(t * frame_duration_ms, 2)
                        ))
                    
                    # Начинаем новый символ
                    current_char = char
                    char_start = t
        
        # Закрываем последний символ
        if current_char is not None and char_start is not None:
            char_timestamps.append(CharTimestamp(
                char=current_char,
                start_ms=round(char_start * frame_duration_ms, 2),
                end_ms=round((T-1) * frame_duration_ms, 2)
            ))
        
        return char_timestamps

class GreedyCTCDecoderTS:
    """
    Жадный CTC‑декодер, который не только возвращает текст,
    но и тайм‑метки для каждого символа.
    frame_duration_ms задаётся извне (у нас 30 мс/кадр).
    """

    def __init__(self, frame_duration_ms: float = 30.0):
        self.frame_ms = frame_duration_ms
        self._prev_id = BLANK_TOKEN_ID          # нужен, чтобы склеивать чанки
        self._open_start: Optional[int] = None  # старт ещё не закрытого символа

    def forward(
        self,
        logprobs: np.ndarray,
        global_frame_offset: int
    ) -> Tuple[str, List[CharTimestamp]]:
        ids = logprobs.argmax(axis=-1)                # (T,)
        text_parts: List[str] = []
        ts_list: List[CharTimestamp] = []

        for local_t, curr_id in enumerate(ids):
            is_changed = (curr_id != self._prev_id)

            # Закрываем предыдущий символ (если был и сменился/стал blank)
            if is_changed and self._prev_id != BLANK_TOKEN_ID:
                end_frame = global_frame_offset + local_t
                ts_list.append(
                    CharTimestamp(
                        char=LABELS[self._prev_id],
                        start_ms=round(self._open_start * self.frame_ms, 2),
                        end_ms=round(end_frame * self.frame_ms, 2)
                    )
                )
                text_parts.append(LABELS[self._prev_id])
                self._open_start = None

            # Открываем новый символ
            if is_changed and curr_id != BLANK_TOKEN_ID:
                self._open_start = global_frame_offset + local_t

            self._prev_id = curr_id

        return "".join(text_parts), ts_list

    def finish(self, global_frame: int) -> Tuple[str, List[CharTimestamp]]:
        text_parts, ts_list = "", []
        if self._prev_id != BLANK_TOKEN_ID and self._open_start is not None:
            ts_list.append(
                CharTimestamp(
                    char=LABELS[self._prev_id],
                    start_ms=round(self._open_start * self.frame_ms, 2),
                    end_ms=round(global_frame * self.frame_ms, 2)
                )
            )
            text_parts = LABELS[self._prev_id]
        # сбрасываем, чтобы можно было заново использовать экземпляр
        self._prev_id = BLANK_TOKEN_ID
        self._open_start = None
        return text_parts, ts_list


class PyctcdecodeTimestampDecoder:
    """
    Декодер, который использует pyctcdecode для лучшего текста,
    а затем делает force alignment для получения таймметок.
    """

    def __init__(self, frame_duration_ms: float = 30.0):
        self.frame_ms = frame_duration_ms
        self.aligner = CTCAlignment()
        
        # Создаем vocab list для pyctcdecode
        vocab_list = list(LABELS) + ["<blank>"]
        
        # Инициализируем pyctcdecode декодер если доступен
        if PYCTCDECODE_AVAILABLE:
            try:
                # Используем build_ctcdecoder как в decoder.py
                self.beam_decoder = build_ctcdecoder(
                    labels=list(LABELS),
                    kenlm_model_path=None,  # Без языковой модели
                    alpha=0.4,  # Как в оригинальном коде
                    beta=0.9    # Как в оригинальном коде
                )
                self.use_pyctcdecode = True
#                print("Using pyctcdecode with beam search (no language model)")
            except Exception as e:
                print(f"Failed to initialize pyctcdecode: {e}")
                self.use_pyctcdecode = False
        else:
            self.use_pyctcdecode = False
        
        # Fallback на greedy декодер
        if not self.use_pyctcdecode:
            self.greedy_decoder = GreedyCTCDecoderTS(frame_duration_ms)
            print("Using greedy CTC decoder")
        
        # Состояние для склеивания чанков (только для greedy)
        self._accumulated_logprobs = []
        self._global_frame_offset = 0

    def forward(self, logprobs: np.ndarray, global_frame_offset: int) -> Tuple[str, List[CharTimestamp]]:
        """Обрабатывает чанк логпробабилити"""
        
        if self.use_pyctcdecode:
            # Накапливаем логпробы для последующего decode при finish()
            self._accumulated_logprobs.append(logprobs)
            self._global_frame_offset = global_frame_offset
            return "", []  # Возвращаем пустые результаты, всё будет в finish()
        else:
            # Используем greedy декодер как раньше
            return self.greedy_decoder.forward(logprobs, global_frame_offset)

    def finish(self, global_frame: int) -> Tuple[str, List[CharTimestamp]]:
        """Финализирует декодирование"""
        
        if self.use_pyctcdecode and self._accumulated_logprobs:
            # Объединяем все накопленные логпробы
            all_logprobs = np.concatenate(self._accumulated_logprobs, axis=0)
            
            # Декодируем с помощью pyctcdecode
            try:
                decoded_text = self.beam_decoder.decode(all_logprobs)  # Transpose для (vocab, time)
                
                # Делаем force alignment для получения таймметок
                char_timestamps = self.aligner.force_align(all_logprobs, decoded_text, self.frame_ms)
                
                # Очищаем накопленные данные
                self._accumulated_logprobs = []
                
                return decoded_text, char_timestamps
                
            except Exception as e:
                print(f"Error in pyctcdecode: {e}, falling back to greedy")
                # Fallback на greedy декодирование
                greedy_decoder = GreedyCTCDecoderTS(self.frame_ms)
                text_parts = []
                char_timestamps = []
                
                frame_offset = 0
                for logprobs_chunk in self._accumulated_logprobs:
                    text_chunk, ts_chunk = greedy_decoder.forward(logprobs_chunk, frame_offset)
                    text_parts.append(text_chunk)
                    char_timestamps.extend(ts_chunk)
                    frame_offset += logprobs_chunk.shape[0]
                
                final_text, final_ts = greedy_decoder.finish(frame_offset)
                text_parts.append(final_text)
                char_timestamps.extend(final_ts)
                
                self._accumulated_logprobs = []
                return "".join(text_parts), char_timestamps
        else:
            # Используем greedy декодер
            return self.greedy_decoder.finish(global_frame)


class WebRTCVADSegmenter:
    """Класс для разбиения аудио на сегменты с помощью WebRTC VAD"""
    
    def __init__(self, aggressiveness=3, frame_duration_ms=30):
        """
        aggressiveness: 0-3, где 3 - самый агрессивный (удаляет больше тишины)
        frame_duration_ms: длительность фрейма для VAD (10, 20 или 30 мс)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = 8000  # WebRTC VAD работает с 8kHz
        
        # Проверяем поддерживаемые параметры
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
    
    def segment_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Разбивает аудио на сегменты речи с помощью VAD
        Returns: список AudioSegment с временными границами
        """
        print(f"Input audio: shape={audio_data.shape}, dtype={audio_data.dtype}, sample_rate={sample_rate}")
        
        # Убеждаемся, что частота дискретизации поддерживается
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate {sample_rate} not supported by WebRTC VAD. Must be 8000, 16000, 32000, or 48000")
        
        # Конвертируем в int16 для VAD
        if audio_data.dtype == np.int32:
            # Предполагаем, что данные в диапазоне int16, но тип int32
            audio_int16 = np.clip(audio_data, -32768, 32767).astype(np.int16)
        elif audio_data.dtype == np.float32:
            # Конвертируем из float32 [-1.0, 1.0] в int16
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
        
        print(f"Converted audio: shape={audio_int16.shape}, dtype={audio_int16.dtype}")
        print(f"Value range: [{np.min(audio_int16)}, {np.max(audio_int16)}]")
        
        # Вычисляем размер фрейма в сэмплах
        frame_length = int(sample_rate * (self.frame_duration_ms / 1000.0))
        print(f"Frame length: {frame_length} samples ({self.frame_duration_ms}ms)")
        
        segments = []
        voiced_frames = []
        
        # Анализируем каждый фрейм
        offset = 0
        frame_count = 0
        
        while offset + frame_length <= len(audio_int16):
            # Извлекаем фрейм
            frame = audio_int16[offset:offset + frame_length]
            
            try:
                # Конвертируем в байты для webrtcvad
                frame_bytes = frame.tobytes()
                
                # Проверяем размер фрейма в байтах
                expected_bytes = frame_length * 2  # 2 bytes per int16 sample
                if len(frame_bytes) != expected_bytes:
                    print(f"Warning: frame {frame_count} has {len(frame_bytes)} bytes, expected {expected_bytes}")
                
                # Проверяем с помощью VAD
                is_speech = self.vad.is_speech(frame_bytes, sample_rate)
                voiced_frames.append((offset, offset + frame_length, is_speech))
                
                frame_count += 1
#                if frame_count % 100 == 0:  # Прогресс каждые 100 фреймов
#                    print(f"Processed {frame_count} frames...")
                
            except Exception as e:
                print(f"Error processing frame {frame_count} at offset {offset}: {e}")
                print(f"Frame shape: {frame.shape}, Frame bytes length: {len(frame.tobytes())}")
                # Пропускаем проблемный фрейм и продолжаем
                voiced_frames.append((offset, offset + frame_length, False))
            
            offset += frame_length
        
        print(f"Total frames processed: {len(voiced_frames)}")
        
        # Группируем соседние речевые фреймы
        if not voiced_frames:
            print("No frames were processed")
            return segments
        
        # Находим начало и конец речевых сегментов
        current_segment_start = None
        speech_frames = sum(1 for _, _, is_speech in voiced_frames if is_speech)
        print(f"Speech frames: {speech_frames}/{len(voiced_frames)} ({speech_frames/len(voiced_frames)*100:.1f}%)")
        
        for offset_start, offset_end, is_speech in voiced_frames:
            if is_speech and current_segment_start is None:
                # Начало нового сегмента
                current_segment_start = offset_start
            elif not is_speech and current_segment_start is not None:
                # Конец текущего сегмента
                start_ms = (current_segment_start / sample_rate) * 1000
                end_ms = (offset_start / sample_rate) * 1000
                
                # Извлекаем аудио сегмент (возвращаем в исходном формате)
                segment_audio = audio_data[current_segment_start:offset_start]
                
                if len(segment_audio) > 0:  # Проверяем, что сегмент не пустой
                    segments.append(AudioSegment(
                        audio_data=segment_audio,
                        start_ms=start_ms,
                        end_ms=end_ms
                    ))
#                    print(f"Added segment: {start_ms:.0f}-{end_ms:.0f}ms ({len(segment_audio)} samples)")
                
                current_segment_start = None
        
        # Если последний сегмент не был закрыт
        if current_segment_start is not None:
            start_ms = (current_segment_start / sample_rate) * 1000
            end_ms = (len(audio_data) / sample_rate) * 1000
            
            segment_audio = audio_data[current_segment_start:]
            if len(segment_audio) > 0:
                segments.append(AudioSegment(
                    audio_data=segment_audio,
                    start_ms=start_ms,
                    end_ms=end_ms
                ))
                print(f"Added final segment: {start_ms:.0f}-{end_ms:.0f}ms ({len(segment_audio)} samples)")
        
        # Если не нашли сегментов речи, возвращаем всё аудио как один сегмент
        if not segments:
            print("No speech segments found, using entire audio as single segment")
            segments.append(AudioSegment(
                audio_data=audio_data,
                start_ms=0,
                end_ms=(len(audio_data) / sample_rate) * 1000
            ))
        
        return segments


# ────────────────────────────────────────────────────────────────────
# АСИНХРОННАЯ ВЕРСИЯ TritonASRClient с параллельной обработкой VAD сегментов
# ────────────────────────────────────────────────────────────────────
class AsyncTritonASRClient:
    CHUNK_SIZE = 2400        # аудио‑сэмплов на вход модели
    FRAMES_PER_CHUNK = 10    # модель выдаёт 10 лог‑кадров
    FRAME_MS = CHUNK_SIZE / FRAMES_PER_CHUNK / 8000 * 1000.0  # 30 мс
    LOOKAHEAD_FRAMES  = 2
    PADDING_SAMPLES   = 2400
    OFFSET_MS = PADDING_SAMPLES / 8000 * 1000.0 + LOOKAHEAD_FRAMES * FRAME_MS

    def __init__(self,
                 triton_url: str = "localhost:8001",
                 model_name: str = "streaming_acoustic",
                 use_pyctcdecode: bool = True,
                 max_parallel_chunks: int = 8):
        """
        max_parallel_chunks: максимальное количество параллельных запросов к GPU
        """
        self.client = grpcclient_aio.InferenceServerClient(url=triton_url, verbose=False)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_parallel_chunks)  # Ограничение параллельности

        # Выбираем декодер
        if use_pyctcdecode and PYCTCDECODE_AVAILABLE:
            self.decoder_class = PyctcdecodeTimestampDecoder
            self.decoder_kwargs = {'frame_duration_ms': self.FRAME_MS}
            print("Using PyctcdecodeTimestampDecoder")
        else:
            self.decoder_class = GreedyCTCDecoderTS
            self.decoder_kwargs = {'frame_duration_ms': self.FRAME_MS}
            print("Using GreedyCTCDecoderTS")

        self.logprob_splitter = StreamingLogprobSplitter()
        self.vad_segmenter = WebRTCVADSegmenter(aggressiveness=1)
        print(f"AsyncTritonASRClient initialized. Frame = {self.FRAME_MS:.1f} ms, max_parallel = {max_parallel_chunks}")

    async def _check_model_ready(self):
        """Асинхронная проверка готовности модели"""
        ready = await self.client.is_model_ready(self.model_name)
        if not ready:
            raise RuntimeError(f"Model {self.model_name} not ready")

    def _apply_offset(self, char_ts, phrase_ts):
        """Применяет временной офсет к timestamp'ам"""
        for ct in char_ts:
            ct.start_ms -= self.OFFSET_MS
            ct.end_ms   -= self.OFFSET_MS
        for pt in phrase_ts:
            pt.start_ms -= self.OFFSET_MS
            pt.end_ms   -= self.OFFSET_MS

    def _finalize_phrase(self, txt, char_ts_in_phrase):
        """Создает PhraseTimestamp из текста и символьных timestamp'ов"""
        if not char_ts_in_phrase:
            return None
        return PhraseTimestamp(
            text      = txt.strip(),
            start_ms  = char_ts_in_phrase[0].start_ms,
            end_ms    = char_ts_in_phrase[-1].end_ms
        )

    async def _infer_chunk_async(self, signal_int32, state_prev):
        if state_prev is None:
            state_prev = np.zeros((1, STATE_SIZE), dtype=np.float16)
        elif state_prev.ndim == 1:
            state_prev = state_prev.reshape(1, -1)

        inputs = [
            grpcclient.InferInput("signal", signal_int32.shape, "INT32"),
            grpcclient.InferInput("state",  state_prev.shape,   "FP16")
        ]
        inputs[0].set_data_from_numpy(signal_int32)
        inputs[1].set_data_from_numpy(state_prev)

        # — единое, корректно сформированное описание выходов —
        requested_outputs = [
            grpcclient.InferRequestedOutput("logprobs"),
            grpcclient.InferRequestedOutput("state_next")
        ]

        # защитный тайм-аут, чтобы RPC не повисал навечно
        try:
            resp = await asyncio.wait_for(
                self.client.infer(
                    self.model_name,
                    inputs=inputs,
                    outputs=requested_outputs    # ← никакого Ellipsis!
                ),
                timeout=5.0                      # сек; подберите под свою сеть/GPU
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Triton RPC timeout (5 s) — сегмент слишком длинный или GPU перегружен")

        logprobs   = resp.as_numpy("logprobs")[0]
        state_next = resp.as_numpy("state_next")
        return logprobs, state_next


    async def _transcribe_segment_async(self, audio_int32: np.ndarray) -> Tuple[str, List[CharTimestamp], List[PhraseTimestamp]]:
        """
        Асинхронная транскрипция одного сегмента аудио
        Возвращает текст и таймметки относительно начала сегмента
        """
        # ---- подготовка аудио, разбиение на чанки ---------------
        pad = (-len(audio_int32)) % self.CHUNK_SIZE
        audio_int32 = np.pad(audio_int32, (0, pad))
        chunks = audio_int32.reshape(-1, self.CHUNK_SIZE)

        global_frame = 0
        model_state = None
        splitter_state = None
        all_text_parts, all_char_ts, phrase_ts = [], [], []

        # Создаем новый декодер и сплиттер для каждого сегмента
        decoder = self.decoder_class(**self.decoder_kwargs)
        splitter = StreamingLogprobSplitter()

        for is_last, chunk in zip(
                [False]* (len(chunks)-1) + [True],      # флаг последнего чанка
                chunks):

            # --- асинхронный инференс через Triton ---------------------------
            logprobs, model_state = await self._infer_chunk_async(
                chunk.reshape(1, self.CHUNK_SIZE, 1),
                model_state)

            # --- передаём в логпроб‑сплиттер ---------------------
            phrases, splitter_state = splitter.forward(
                logprobs, splitter_state, is_last=is_last)

            # --- для каждого законченного фрагмента --------------
            for ph in phrases:
                txt, cts_part = decoder.forward(ph.logprobs, ph.start_frame)
                tail_txt, tail_ts = decoder.finish(ph.end_frame)
                txt += tail_txt
                cts_part.extend(tail_ts)

                # --- смещаем на OFFSET_MS ------------------------------
                self._apply_offset(cts_part, [])           # только символы

                if txt.strip():
                    all_text_parts.append(txt)
                    all_char_ts.extend(cts_part)

                    phrase = self._finalize_phrase(txt, cts_part)
                    if phrase:
                        phrase_ts.append(phrase)

            self._apply_offset([], phrase_ts)

            # обновляем глобальный счётчик кадров
            global_frame += logprobs.shape[0]

        transcription = " ".join(all_text_parts).strip()
        return transcription, all_char_ts, phrase_ts

    async def transcribe_single_wav_async(self, wav_file: str) -> Tuple[str, List[CharTimestamp], List[PhraseTimestamp]]:
        """
        Асинхронная транскрипция одного WAV файла с разбиением на сегменты через VAD
        и параллельной обработкой сегментов
        """
        # Проверяем готовность модели
        await self._check_model_ready()
        
        # Читаем аудио файл
        audio_data, sample_rate = sf.read(wav_file)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print(f"Converted {wav_file} from stereo to mono")
        
        # Convert to float32 for processing
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Resample to 8kHz if needed (T-One expects 8kHz)
        target_sr = 8000
        if sample_rate != target_sr:
            print(f"Resampling {wav_file} from {sample_rate}Hz to {target_sr}Hz")
            audio_data = resample_audio(audio_data, sample_rate, target_sr)
            sample_rate = target_sr
        
        # Convert to int32 format as required by T-One
        audio_int32 = convert_float32_to_int32(audio_data)
        
        # Разбиваем аудио на сегменты с помощью VAD
        print(f"Segmenting audio with WebRTC VAD...")
        segments = self.vad_segmenter.segment_audio(audio_int32, sample_rate)
        print(f"Found {len(segments)} speech segments")
        
        # ← ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА СЕГМЕНТОВ ←
        print(f"Starting parallel processing of {len(segments)} segments...")
        
        segment_results = []
        for seg in segments:
            segment_results.append(
                await self._transcribe_segment_async(seg.audio_data)
            )
        
        # Объединяем результаты с учетом временных офсетов сегментов
        all_text_parts = []
        all_char_ts = []
        all_phrase_ts = []
        
        for i, (segment, (segment_text, segment_char_ts, segment_phrase_ts)) in enumerate(zip(segments, segment_results)):
            print(f"Segment {i+1}/{len(segments)} result: '{segment_text[:50]}...' ({len(segment_char_ts)} chars, {len(segment_phrase_ts)} phrases)")
            
            if segment_text.strip():
                all_text_parts.append(segment_text)
                
                # Корректируем таймметки на offset сегмента
                for char_ts in segment_char_ts:
                    char_ts.start_ms += segment.start_ms
                    char_ts.end_ms += segment.start_ms
                
                for phrase_ts in segment_phrase_ts:
                    phrase_ts.start_ms += segment.start_ms
                    phrase_ts.end_ms += segment.start_ms
                
                all_char_ts.extend(segment_char_ts)
                all_phrase_ts.extend(segment_phrase_ts)
        
        # Объединяем результаты
        final_transcription = " ".join(all_text_parts).strip()
        
        return final_transcription, all_char_ts, all_phrase_ts

    async def close(self):
        """Закрывает асинхронное соединение с Triton"""
        await self.client.close()


# ────────────────────────────────────────────────────────────────────
# Вспомогательные функции (остаются без изменений)
# ────────────────────────────────────────────────────────────────────
def calculate_wer(hypothesis, reference):
    print(f"hyp: {hypothesis}")
    print(f"ref: {reference}")
    return jiwer.wer(re.sub(r'[",.!?]', '', reference.strip()).lower(), 
                     re.sub(r'[",.!?]', '', hypothesis.strip()).lower())

def convert_float32_to_int32(audio_data):
    """Convert float32 audio [-1.0, 1.0] to int32 format with int16 range for T-One"""
    # Clip values to [-1.0, 1.0] range to avoid overflow
    audio_data = np.clip(audio_data, -1.0, 1.0)
    # Convert to int16 range but keep int32 dtype (T-One expects int32 dtype with int16 range)
    audio_data = (audio_data * 32767).astype(np.int32)
    return audio_data

def resample_audio(audio_data, original_sr, target_sr=8000):
    """Resample audio to target sample rate if needed"""
    if original_sr != target_sr:
        # Calculate the number of samples for the target sample rate
        num_samples = int(len(audio_data) * target_sr / original_sr)
        audio_data = signal.resample(audio_data, num_samples)
    return audio_data


# ────────────────────────────────────────────────────────────────────
# АСИНХРОННАЯ ФУНКЦИЯ ОБРАБОТКИ ФАЙЛОВ
# ────────────────────────────────────────────────────────────────────
async def process_audio_files_async(use_pyctcdecode: bool = True, max_parallel_chunks: int = 8):
    """
    Асинхронная обработка аудио файлов с параллельной транскрипцией VAD сегментов
    """
    # Start total processing time
    total_start_time = time.time()
    
    # Initialize async Triton ASR client
    print("Initializing async Triton ASR gRPC client...")
    try:
        asr_client = AsyncTritonASRClient(
            triton_url="localhost:8001",  # gRPC порт по умолчанию
            model_name="streaming_acoustic",
            use_pyctcdecode=use_pyctcdecode,
            max_parallel_chunks=max_parallel_chunks
        )
        print("Async Triton ASR gRPC client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize async Triton gRPC client: {e}")
        return
    
    try:
        # Read CSV using pandas
        df = pd.read_csv('aud1.csv', header=None, names=['filename', 'reference_text'])
        parsed_data = df.values.tolist()
        
        total_wer = 0
        processed_files = 0
        total_transcription_time = 0
        results = []
        errors = []
        
        for row in parsed_data:
            wav_file = os.path.join("audio_for_annotation", row[0])
            reference_text = row[1]
            
            try:
                # Check if file exists
                if not os.path.exists(wav_file):
                    errors.append(f"File not found: {wav_file}")
                    continue
                
                print(f"\nProcessing: {wav_file}")
                
                # Start timing transcription
                transcription_start_time = time.time()
                
                # Get transcription via async Triton gRPC with parallel VAD segment processing
                hypothesis_text, char_ts, phrase_ts = await asr_client.transcribe_single_wav_async(wav_file)
                
                print("Phrase timestamps:")
                for i, phrase in enumerate(phrase_ts):
                    print(f"  {i+1}: '{phrase.text}' ({phrase.start_ms:.0f}-{phrase.end_ms:.0f}ms)")
                
                # End timing transcription
                transcription_end_time = time.time()
                transcription_time = transcription_end_time - transcription_start_time
                total_transcription_time += transcription_time
                
                # Calculate WER for this file
                if hypothesis_text.strip():  # Only calculate WER if we got a transcription
                    wer = calculate_wer(hypothesis_text, reference_text)
                else:
                    wer = 1.0  # If no transcription, WER is 100%
                    print(f"Warning: Empty transcription for {wav_file}")
                
                results.append({
                    'file': wav_file,
                    'reference': reference_text,
                    'hypothesis': hypothesis_text,
                    'wer': wer,
                    'transcription_time_seconds': round(transcription_time, 2),
                    'num_segments': len([p for p in phrase_ts if p.text.strip()]),
                    'total_speech_duration_ms': sum([p.end_ms - p.start_ms for p in phrase_ts])
                })
                
                total_wer += wer
                processed_files += 1
                print(f"Processed {wav_file}: WER = {wer:.4f}, Time = {transcription_time:.2f}s, Segments = {len(phrase_ts)}")
                
            except Exception as e:
                errors.append(f"Error processing {wav_file}: {str(e)}")
                print(f"Skipping {wav_file} due to error: {str(e)}")
                continue
    
    finally:
        # Закрываем соединение с Triton
        await asr_client.close()
    
    # End total processing time
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    # Print results
    if processed_files > 0:
        average_wer = total_wer / processed_files
        average_transcription_time = total_transcription_time / processed_files
        
        print(f"\nOverall Results (Async with Parallel VAD Segments):")
        print(f"Total files processed successfully: {processed_files}")
        print(f"Average WER: {average_wer:.4f}")
        print(f"\nTiming Statistics:")
        print(f"Total transcription time: {total_transcription_time:.2f}s")
        print(f"Average transcription time per file: {average_transcription_time:.2f}s")
        print(f"Total processing time (including overhead): {total_processing_time:.2f}s")
        print(f"Transcription efficiency: {(total_transcription_time/total_processing_time)*100:.1f}% of total time")
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('async_triton_grpc_asr_vad_results.csv', index=False)
        print("\nDetailed results saved to async_triton_grpc_asr_vad_results.csv")
        
        # Save timing summary
        timing_summary = {
            'total_files': processed_files,
            'total_transcription_time_seconds': round(total_transcription_time, 2),
            'average_transcription_time_seconds': round(average_transcription_time, 2),
            'total_processing_time_seconds': round(total_processing_time, 2),
            'transcription_efficiency_percent': round((total_transcription_time/total_processing_time)*100, 1),
            'max_parallel_chunks': max_parallel_chunks
        }
        
        timing_df = pd.DataFrame([timing_summary])
        timing_df.to_csv('async_triton_grpc_vad_timing_summary.csv', index=False)
        print("Timing summary saved to async_triton_grpc_vad_timing_summary.csv")
        
    else:
        print("No files were successfully processed")
    
    # Print error summary
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)
        
        # Save errors to file
        with open('async_triton_grpc_vad_processing_errors.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(errors))
        print("\nErrors have been saved to async_triton_grpc_vad_processing_errors.txt")


# ────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ ДЛЯ ЗАПУСКА
# ────────────────────────────────────────────────────────────────────
async def main():
    """Главная асинхронная функция"""
    print("Starting async ASR processing with parallel VAD segment transcription...")
    
    # Параметры для настройки производительности:
    # use_pyctcdecode=True - использовать pyctcdecode если доступен (лучшее качество)
    # max_parallel_chunks=8 - максимальное количество одновременных запросов к GPU
    #                         Увеличьте для более мощного GPU, уменьшите при нехватке VRAM
    await process_audio_files_async(
        use_pyctcdecode=True, 
        max_parallel_chunks=16  # ← ЗДЕСЬ задается количество параллельных обработок
    )


if __name__ == "__main__":
    # Установите зависимости:
    # pip install tritonclient[grpc] soundfile jiwer pandas scipy numpy webrtcvad
    # 
    # Для использования pyctcdecode (опционально, для лучшего качества):
    # pip install pyctcdecode
    
    # Игнорируем предупреждения gRPC
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Запуск асинхронной обработки
    asyncio.run(main())