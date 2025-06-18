@echo off
echo ================================
echo   Whisper Transcriber Builder
echo ================================

:: Проверка наличия виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    
    echo Installing dependencies...
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install openai-whisper
    pip install customtkinter
    pip install pyinstaller
    pip install tiktoken
    pip install regex
    pip install ftfy
    pip install more-itertools
) else (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

:: Проверка наличия основного файла
if not exist "transcriber_app.py" (
    echo ERROR: transcriber_app.py not found!
    echo Please ensure your main Python file is named transcriber_app.py
    pause
    exit /b 1
)

:: Проверка наличия FFmpeg
if not exist "bin\ffmpeg.exe" (
    echo ERROR: FFmpeg not found in bin directory!
    echo Please ensure you have:
    echo - bin\ffmpeg.exe
    echo - bin\ffprobe.exe
    echo - All required DLL files
    pause
    exit /b 1
)

:: Очистка предыдущих сборок
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "__pycache__" rmdir /s /q "__pycache__"

:: Сборка
echo Starting PyInstaller build...
pyinstaller --clean whisper_transcriber.spec

:: Проверка результата
if exist "dist\WhisperTranscriber\WhisperTranscriber.exe" (
    echo.
    echo ================================
    echo   BUILD SUCCESSFUL!
    echo ================================
    echo.
    echo Executable created: dist\WhisperTranscriber\WhisperTranscriber.exe
    echo.
    echo You can now copy the entire dist\WhisperTranscriber folder
    echo to any Windows computer and run the application.
    echo.
    
    :: Показываем размер
    for /f %%i in ('dir "dist\WhisperTranscriber" /s /-c ^| find "bytes"') do echo Total size: %%i
    
    echo.
    echo Opening build directory...
    explorer "dist\WhisperTranscriber"
    
) else (
    echo.
    echo ================================
    echo   BUILD FAILED!
    echo ================================
    echo.
    echo Check the output above for errors.
    echo Common issues:
    echo - Missing dependencies
    echo - Incorrect file paths
    echo - Insufficient disk space
)

echo.
pause