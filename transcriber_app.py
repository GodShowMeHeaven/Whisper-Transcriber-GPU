import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
import whisper
import threading
import os
import sys
from io import StringIO
import contextlib
import torch
import time
import logging
import warnings
import re
import textwrap
import urllib.request

class WhisperLogHandler(logging.Handler):
    """Кастомный обработчик логов для Whisper"""
    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback
        
    def emit(self, record):
        log_message = self.format(record)
        self.update_callback(log_message + '\n')

class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcriber - GPU/CPU")
        self.root.geometry("1000x800")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.suppress_warnings()

        self.use_gpu = self.check_gpu_availability()
        if not self.use_gpu:
            messagebox.showwarning("Внимание", 
                                "GPU недоступен. Транскрибация будет выполнена на CPU.\n"
                                "Производительность может быть ниже.")

        # Установка пути к ffmpeg
        import os
        import sys
        ffmpeg_path = os.path.join(os.path.dirname(sys.executable), "bin")
        if os.path.exists(ffmpeg_path):
            os.environ["PATH"] += os.pathsep + ffmpeg_path

        self.model = None
        self.filename = ""
        self.is_transcribing = False
        self.last_result = None
        self._last_processing_time = 0
        self.selected_model = tk.StringVar(value="large-v2")

        self.create_widgets()
        self.load_model()

    def suppress_warnings(self):
        warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
        warnings.filterwarnings("ignore", message=".*Triton kernels.*")
        warnings.filterwarnings("ignore", message=".*DTW implementation.*")

    def check_gpu_availability(self):
        try:
            if not torch.cuda.is_available():
                return False
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"GPU проверка не прошла: {e}")
            return False

    def get_gpu_info(self):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                'name': device_name,
                'memory_allocated': memory_allocated,
                'memory_reserved': memory_reserved,
                'memory_total': memory_total,
                'device_count': torch.cuda.device_count()
            }
        return None

    def setup_whisper_logging(self):
        self.whisper_log_handler = WhisperLogHandler(self.update_log_safe)
        self.whisper_log_handler.setLevel(logging.DEBUG)
        whisper_logger = logging.getLogger('whisper')
        whisper_logger.setLevel(logging.DEBUG)
        whisper_logger.addHandler(self.whisper_log_handler)

    def cleanup_whisper_logging(self):
        try:
            whisper_logger = logging.getLogger('whisper')
            if hasattr(self, 'whisper_log_handler'):
                whisper_logger.removeHandler(self.whisper_log_handler)
        except:
            pass

    @contextlib.contextmanager
    def capture_whisper_output(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        class WhisperOutputCapture:
            def __init__(self, update_func):
                self.update_func = update_func
                self.original_stdout = old_stdout
                self.original_stderr = old_stderr
                
            def write(self, text):
                self.original_stdout.write(text)
                self.original_stdout.flush()
                if text.strip():
                    self.update_func(text)
                    
            def flush(self):
                self.original_stdout.flush()
        
        capture = WhisperOutputCapture(self.update_log_safe)
        
        try:
            sys.stdout = capture
            sys.stderr = capture
            yield capture
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def format_segments_as_lines(self, segments, max_line_length=80):
        if not segments:
            return ""
        
        formatted_lines = []
        
        for segment in segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            if not text:
                continue
            
            start_time = f"{int(start//60):02d}:{start%60:06.3f}"
            end_time = f"{int(end//60):02d}:{end%60:06.3f}"
            timestamp = f"[{start_time} --> {end_time}]"
            
            available_width = max_line_length - len(timestamp) - 1
            if available_width > 20 and len(text) > available_width:
                wrapped_text = textwrap.fill(text, width=available_width)
                text_lines = wrapped_text.split('\n')
                for i, line in enumerate(text_lines):
                    if i == 0:
                        formatted_lines.append(f"{timestamp} {line}")
                    else:
                        formatted_lines.append(f"{' ' * len(timestamp)} {line}")
            else:
                formatted_lines.append(f"{timestamp} {text}")
        
        return '\n'.join(formatted_lines)

    def format_text_with_line_breaks(self, text, max_line_length=80):
        if not text or not text.strip():
            return text
        
        text = ' '.join(text.split())
        
        if len(text) <= max_line_length:
            return text
        
        sentences = re.split(r'([.!?]+\s*)', text)
        formatted_lines = []
        current_line = ""
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            if not sentence:
                i += 1
                continue
            
            if sentence in '.!?' or (len(sentence) <= 3 and re.match(r'[.!?]+', sentence)):
                if current_line:
                    current_line += sentence
                i += 1
                continue
            
            test_line = current_line + " " + sentence if current_line else sentence
            
            if len(test_line) <= max_line_length:
                current_line = test_line
            else:
                if current_line:
                    formatted_lines.append(current_line)
                
                if len(sentence) > max_line_length:
                    wrapped = textwrap.fill(sentence, width=max_line_length)
                    formatted_lines.extend(wrapped.split('\n'))
                    current_line = ""
                else:
                    current_line = sentence
            
            i += 1
        
        if current_line:
            formatted_lines.append(current_line)
        
        return '\n'.join(formatted_lines)

    def update_output_safe(self, text):
        def update():
            try:
                self.output.insert("end", text)
                self.output.see("end")
            except tk.TclError:
                pass
        
        try:
            self.root.after(0, update)
        except tk.TclError:
            pass

    def update_log_safe(self, text):
        def update():
            try:
                if "Начинаю обработку" in text or not self.is_transcribing:
                    self.log_output.insert("end", text)
                    self.log_output.see("end")
                elif self.is_transcribing:
                    self.output.insert("end", text)
                    self.output.see("end")
            except tk.TclError:
                pass
        
        try:
            self.root.after(0, update)
        except tk.TclError:
            pass

    def create_widgets(self):
        # Главный фрейм
        main_frame = ctk.CTkFrame(self.root, corner_radius=0)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Информация о GPU/CPU с использованием grid для центрирования
        device_info = self.get_gpu_info() if self.use_gpu else {"name": "CPU (без GPU)"}
        device_text = f"🚀 Устройство: {device_info['name']}"
        if self.use_gpu and device_info['device_count'] > 1:
            device_text += f" (доступно {device_info['device_count']} GPU)"
        memory_text = f"💾 Память: {device_info.get('memory_total', 0):.1f} GB" if self.use_gpu else ""

        # Настройка grid для меток
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure((0, 1, 2), weight=1)

        device_label = ctk.CTkLabel(main_frame, text=device_text, font=ctk.CTkFont("Arial", 14, "bold"), 
                                   text_color="#00FF00" if self.use_gpu else "#FF4500")
        device_label.grid(row=0, column=0, pady=5)

        if memory_text:
            memory_label = ctk.CTkLabel(main_frame, text=memory_text, font=ctk.CTkFont("Arial", 12), text_color="#1E90FF")
            memory_label.grid(row=1, column=0, pady=2)

        model_label = ctk.CTkLabel(main_frame, text="Модель:", font=ctk.CTkFont("Arial", 14))
        model_label.grid(row=2, column=0, pady=2)

        # Выпадающий список для выбора модели
        self.model_combo = ctk.CTkComboBox(main_frame, variable=self.selected_model,
                                          values=["base", "small", "medium", "large-v2", "large-v3"],
                                          font=ctk.CTkFont("Arial", 12), width=150)
        self.model_combo.grid(row=3, column=0, pady=5, padx=10)

        # Фрейм для выбора файла и транскрибации
        control_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        control_frame.grid(row=4, column=0, pady=10, sticky="nsew")
        control_frame.grid_columnconfigure(0, weight=1)

        self.label = ctk.CTkLabel(control_frame, text="Файл не выбран", font=ctk.CTkFont("Arial", 12), wraplength=800)
        self.label.pack(pady=5, padx=10)

        select_button = ctk.CTkButton(control_frame, text="📁 Выбрать аудио/видео", command=self.select_file,
                                     font=ctk.CTkFont("Arial", 14), width=200)
        select_button.pack(pady=5)

        self.transcribe_btn = ctk.CTkButton(control_frame, text=f"🚀 Начать транскрибацию ({'GPU' if self.use_gpu else 'CPU'})", 
                                           command=self.start_transcription,
                                           font=ctk.CTkFont("Arial", 14, "bold"), width=250, 
                                           fg_color="#4CAF50", text_color_disabled="#000000")
        self.transcribe_btn.pack(pady=5)

        # Вкладки для результата и логов
        notebook = ctk.CTkTabview(main_frame, height=400)
        notebook.grid(row=5, column=0, pady=10, sticky="nsew")
        main_frame.grid_rowconfigure(5, weight=1)

        # Вкладка результата
        result_tab = notebook.add("📄 Результат")
        result_label = ctk.CTkLabel(result_tab, text="💬 Транскрибированный текст:", font=ctk.CTkFont("Arial", 14, "bold"))
        result_label.pack(anchor="w", padx=10, pady=5)

        self.output = ctk.CTkTextbox(result_tab, font=ctk.CTkFont("Consolas", 12), wrap="word", height=300)
        self.output.pack(fill="both", expand=True, padx=10, pady=5)

        # Вкладка логов
        log_tab = notebook.add("📊 Логи")
        log_label = ctk.CTkLabel(log_tab, text="🔍 Подробные логи запуска:", font=ctk.CTkFont("Arial", 14, "bold"))
        log_label.pack(anchor="w", padx=10, pady=5)

        self.log_output = ctk.CTkTextbox(log_tab, font=ctk.CTkFont("Consolas", 11), wrap="word", height=300, fg_color="#2E2E2E")
        self.log_output.pack(fill="both", expand=True, padx=10, pady=5)

        # Фрейм для кнопок управления
        button_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        button_frame.grid(row=6, column=0, pady=5, sticky="nsew")
        button_frame.grid_columnconfigure((0, 1, 2), weight=1)

        save_button = ctk.CTkButton(button_frame, text="💾 Сохранить", command=self.save_result,
                                   font=ctk.CTkFont("Arial", 12), width=120)
        save_button.grid(row=0, column=0, padx=10, pady=5)

        clear_button = ctk.CTkButton(button_frame, text="🗑️ Очистить", command=self.clear_output,
                                    font=ctk.CTkFont("Arial", 12), width=120)
        clear_button.grid(row=0, column=1, padx=10, pady=5)

        copy_button = ctk.CTkButton(button_frame, text="📋 Копировать", command=self.copy_to_clipboard,
                                   font=ctk.CTkFont("Arial", 12), width=120)
        copy_button.grid(row=0, column=2, padx=10, pady=5)

        # Фрейм для настроек форматирования
        settings_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        settings_frame.grid(row=7, column=0, pady=5, sticky="nsew")
        main_frame.grid_rowconfigure(7, weight=1)

        line_length_label = ctk.CTkLabel(settings_frame, text="Длина строки:", font=ctk.CTkFont("Arial", 12))
        line_length_label.pack(side="left", padx=10)

        self.line_length_var = tk.StringVar(value="80")
        length_spinbox = ctk.CTkEntry(settings_frame, textvariable=self.line_length_var, width=60,
                                     font=ctk.CTkFont("Arial", 12))
        length_spinbox.pack(side="left", padx=5)

        format_label = ctk.CTkLabel(settings_frame, text="Формат:", font=ctk.CTkFont("Arial", 12))
        format_label.pack(side="left", padx=20)

        self.format_mode_var = tk.StringVar(value="segments")
        format_combo = ctk.CTkComboBox(settings_frame, variable=self.format_mode_var,
                                      values=["segments", "paragraphs", "continuous"],
                                      font=ctk.CTkFont("Arial", 12), width=150)
        format_combo.pack(side="left", padx=5)

        self.show_timestamps_var = tk.BooleanVar(value=True)
        timestamps_check = ctk.CTkCheckBox(settings_frame, text="Временные метки",
                                          variable=self.show_timestamps_var,
                                          font=ctk.CTkFont("Arial", 12))
        timestamps_check.pack(side="left", padx=20)

        # Кнопка для применения настроек
        apply_settings_btn = ctk.CTkButton(settings_frame, text="Применить настройки", command=self.apply_settings,
                                          font=ctk.CTkFont("Arial", 12), width=120)
        apply_settings_btn.pack(side="left", padx=10)

    def apply_settings(self):
        if self.last_result:
            self.display_result(self.last_result)
            messagebox.showinfo("Готово", "Настройки применены!")

    def copy_to_clipboard(self):
        text = self.output.get("0.0", "end").strip()
        if not text:
            messagebox.showinfo("Пусто", "Нет текста для копирования.")
            return
        
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Готово", "Текст скопирован в буфер обмена!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать: {e}")

    def clear_output(self):
        self.output.delete("0.0", "end")
        self.log_output.delete("0.0", "end")
        self.is_transcribing = False
        self.last_result = None

    def load_model(self):
        threading.Thread(target=self._load_model_thread, daemon=True).start()
    
    def _load_model_thread(self):
        try:
            device_info = self.get_gpu_info() if self.use_gpu else {"name": "CPU"}
            self.update_log_safe("=== 🚀 ИНФОРМАЦИЯ ОБ УСТРОЙСТВЕ ===\n")
            self.update_log_safe(f"Устройство: {device_info['name']}\n")
            if self.use_gpu:
                self.update_log_safe(f"Доступно GPU: {device_info['device_count']}\n")
                self.update_log_safe(f"Общая память: {device_info['memory_total']:.2f} GB\n")
                self.update_log_safe(f"Память выделена: {device_info['memory_allocated']:.2f} GB\n")
                self.update_log_safe(f"Память зарезервирована: {device_info['memory_reserved']:.2f} GB\n")
            self.update_log_safe("=" * 50 + "\n\n")
            
            self.update_log_safe(f"🔄 Загружаю модель Whisper {self.selected_model.get()} на {device_info['name']}...\n")
            
            device = "cuda:0" if self.use_gpu else "cpu"
            self.setup_whisper_logging()
            
            # Проверка интернет-соединения
            try:
                urllib.request.urlopen('https://huggingface.co', timeout=5)
                has_internet = True
            except:
                has_internet = False
            
            if not has_internet:
                raise Exception("Интернет-соединение отсутствует. Подключитесь к интернету для загрузки модели.")
            
            with self.capture_whisper_output():
                self.model = whisper.load_model(self.selected_model.get(), device=device)
            
            if hasattr(self.model, 'device'):
                actual_device = str(self.model.device)
                self.update_log_safe(f"✅ Модель загружена на устройство: {actual_device}\n")
                
                if self.use_gpu and "cuda" not in actual_device.lower():
                    raise Exception(f"Модель загрузилась на {actual_device}, а не на GPU!")
                elif not self.use_gpu and "cpu" not in actual_device.lower():
                    raise Exception(f"Модель загрузилась на {actual_device}, а не на CPU!")
            
            if self.use_gpu:
                gpu_info_after = self.get_gpu_info()
                self.update_log_safe(f"💾 Память GPU после загрузки: {gpu_info_after['memory_allocated']:.2f} GB\n")
            
            self.update_log_safe(f"\n🚀 Модель {self.selected_model.get()} успешно загружена на {device_info['name']}!\n")
            self.update_log_safe("📋 Готов к транскрибации!\n\n")
            
            self.root.after(0, lambda: self.transcribe_btn.configure(
                text=f"🚀 Начать транскрибацию ({'GPU' if self.use_gpu else 'CPU'})", state="normal"))
            
        except Exception as e:
            error_msg = f"❌ КРИТИЧЕСКАЯ ОШИБКА при загрузке модели: {e}\n\n"
            self.update_log_safe(error_msg)
            
            self.update_log_safe("🔧 Возможные причины:\n")
            if not self.use_gpu:
                self.update_log_safe("1. CPU доступен, но модель не загружается\n")
            else:
                self.update_log_safe("1. Не установлен PyTorch с CUDA поддержкой\n")
                self.update_log_safe("2. Устаревшие драйверы NVIDIA\n")
                self.update_log_safe("3. Недостаточно памяти GPU\n")
            self.update_log_safe("4. Отсутствует интернет-соединение для загрузки модели\n\n")
            self.update_log_safe("💡 Для установки PyTorch с CUDA (если требуется GPU):\n")
            self.update_log_safe("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n")
            self.update_log_safe("💡 Для ручной загрузки модели: скачайте с https://huggingface.co/whisper и поместите в ~/.cache/huggingface/hub.\n")
            
            self.root.after(0, lambda: messagebox.showerror("Критическая ошибка", 
                               f"Не удалось загрузить модель:\n{e}\n\n"
                               "Проверьте интернет-соединение или установите модель вручную."))

    def select_file(self):
        filetypes = [
            ("Все поддерживаемые", "*.mp3;*.wav;*.m4a;*.webm;*.ogg;*.flac;*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.flv;*.3gp"),
            ("Аудио файлы", "*.mp3;*.wav;*.m4a;*.webm;*.ogg;*.flac"),
            ("Видео файлы", "*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.flv;*.3gp"),
            ("Все файлы", "*.*")
        ]
        
        self.filename = filedialog.askopenfilename(filetypes=filetypes)
        
        if self.filename:
            filename_display = os.path.basename(self.filename)
            file_size = os.path.getsize(self.filename) / (1024*1024)
            self.label.configure(text=f"✅ Выбран: {filename_display} ({file_size:.1f} MB)")
        else:
            self.label.configure(text="Файл не выбран")

    def start_transcription(self):
        if not self.filename:
            messagebox.showwarning("Внимание", "Сначала выберите аудио или видео файл.")
            return
        
        if not self.model:
            messagebox.showwarning("Внимание", "Модель еще загружается. Подождите.")
            return

        if self.use_gpu and not torch.cuda.is_available():
            messagebox.showerror("Ошибка GPU", "GPU стал недоступен! Переключение на CPU не поддерживается после загрузки.")
            return

        self.output.delete("0.0", "end")
        self.log_output.delete("0.0", "end")
        self.is_transcribing = False
        
        self.transcribe_btn.configure(state="disabled", text=f"⚡ Транскрибация ({'GPU' if self.use_gpu else 'CPU'})...", text_color="#000000")
        
        threading.Thread(target=self.transcribe_audio, daemon=True).start()

    def display_result(self, result):
        device_info = self.get_gpu_info() if self.use_gpu else {"name": "CPU"}
        processing_time = self._last_processing_time
        
        result_header = f"=== ⚡ РЕЗУЛЬТАТ ТРАНСКРИБАЦИИ ===\n"
        result_header += f"🚀 Устройство: {device_info['name']}\n"
        result_header += f"📁 Файл: {os.path.basename(self.filename)}\n"
        result_header += f"⏱️ Время: {processing_time:.1f} секунд\n"
        result_header += f"🌍 Язык: {result.get('language', 'ru')}\n"
        result_header += f"📝 Символов: {len(result['text'])}\n"
        if processing_time > 0:
            result_header += f"🚀 Скорость: {len(result['text'])/processing_time:.0f} символов/сек\n"
        result_header += "=" * 60 + "\n\n"
        
        try:
            max_line_length = int(self.line_length_var.get())
            if max_line_length < 20:
                max_line_length = 80
        except ValueError:
            max_line_length = 80
            
        format_mode = self.format_mode_var.get()
        show_timestamps = self.show_timestamps_var.get()
        
        formatted_text = ""
        if format_mode == "segments" and "segments" in result and result["segments"]:
            if show_timestamps:
                formatted_text = self.format_segments_as_lines(result["segments"], max_line_length)
            else:
                for segment in result["segments"]:
                    text = segment.get("text", "").strip()
                    if text:
                        if len(text) > max_line_length:
                            wrapped_text = textwrap.fill(text, width=max_line_length)
                            formatted_text += wrapped_text + "\n\n"
                        else:
                            formatted_text += text + "\n\n"
                formatted_text = formatted_text.rstrip()
        elif format_mode == "paragraphs" and "segments" in result and result["segments"]:
            paragraphs = []
            current_paragraph = []
            for segment in result["segments"]:
                text = segment.get("text", "").strip()
                if text:
                    current_paragraph.append(text)
                    if (segment.get("end", 0) - segment.get("start", 0) > 2.0 or 
                        text.rstrip().endswith(('.', '!', '?'))):
                        if current_paragraph:
                            paragraph_text = ' '.join(current_paragraph)
                            wrapped_text = textwrap.fill(paragraph_text, width=max_line_length)
                            paragraphs.append(wrapped_text)
                            current_paragraph = []
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                wrapped_text = textwrap.fill(paragraph_text, width=max_line_length)
                paragraphs.append(wrapped_text)
            formatted_text = '\n\n'.join(paragraphs)
        else:
            formatted_text = self.format_text_with_line_breaks(result["text"], max_line_length)
        
        def update_final_result():
            try:
                self.output.delete("0.0", "end")
                self.output.insert("end", result_header + formatted_text)
            except Exception:
                pass
                
        self.root.after(0, update_final_result)

    def transcribe_audio(self):
        try:
            device_info = self.get_gpu_info() if self.use_gpu else {"name": "CPU"}
            file_ext = os.path.splitext(self.filename)[1].lower()
            file_type = "видео" if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.3gp'] else "аудио"
            
            self.update_log_safe(f"🎬 Начинаю обработку {file_type} файла на {device_info['name']}...\n")
            self.update_log_safe(f"📁 Файл: {os.path.basename(self.filename)}\n")
            if self.use_gpu:
                self.update_log_safe(f"🚀 GPU: {device_info['name']}\n")
                self.update_log_safe(f"💾 Память до обработки: {device_info['memory_allocated']:.2f} GB\n")
            self.update_log_safe("=" * 50 + "\n")
            
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            start_time = time.time()
            
            self.is_transcribing = True
            self.update_log_safe(f"Начинаю обработку на {device_info['name']} с моделью {self.selected_model.get()}...\n")
            
            with self.capture_whisper_output():
                result = self.model.transcribe(
                    self.filename,
                    language="ru",
                    task="transcribe",
                    fp16=self.use_gpu,  # FP16 только для GPU
                    verbose=True,
                    word_timestamps=True
                )
            
            processing_time = time.time() - start_time
            self._last_processing_time = processing_time
            self.last_result = result
            
            self.is_transcribing = False
            
            if self.use_gpu:
                gpu_info_after = self.get_gpu_info()
                self.update_log_safe(f"\n✅ Транскрибация завершена за {processing_time:.1f} секунд!\n")
                self.update_log_safe(f"💾 Память {device_info['name']} после: {gpu_info_after['memory_allocated']:.2f} GB\n")
            else:
                self.update_log_safe(f"\n✅ Транскрибация завершена за {processing_time:.1f} секунд!\n")
            self.update_log_safe(f"📝 Обработано символов: {len(result['text'])}\n")
            if processing_time > 0:
                self.update_log_safe(f"🚀 Скорость: {len(result['text'])/processing_time:.0f} символов/сек\n")
            self.update_log_safe("=" * 60 + "\n")
            
            self.display_result(result)
            
            if self.use_gpu:
                torch.cuda.empty_cache()
            
        except Exception as e:
            self.is_transcribing = False
            error_msg = f"❌ Ошибка при транскрибации: {e}\n"
            self.update_log_safe(error_msg)
            
            try:
                if self.use_gpu:
                    torch.cuda.empty_cache()
            except:
                pass
                
            self.root.after(0, lambda: messagebox.showerror("Ошибка транскрибации", str(e)))
        
        finally:
            self.root.after(0, lambda: self.transcribe_btn.configure(
                state="normal", text=f"🚀 Начать транскрибацию ({'GPU' if self.use_gpu else 'CPU'})", text_color="white"))

    def save_result(self):
        text = self.output.get("0.0", "end").strip()
        if not text:
            messagebox.showinfo("Пусто", "Нет результата для сохранения.")
            return
        
        filetypes = [
            ("Текстовые файлы", "*.txt"),
            ("Все файлы", "*.*")
        ]
        
        if self.filename:
            base_name = os.path.splitext(os.path.basename(self.filename))[0]
            initialfile = f"{base_name}_transcript.txt"
        else:
            initialfile = "transcript.txt"
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=filetypes,
            initialfile=initialfile
        )
        
        if save_path and save_path.strip():
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Готово", f"Результат сохранен в:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл:\n{e}")

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    root = ctk.CTk()
    app = WhisperApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()