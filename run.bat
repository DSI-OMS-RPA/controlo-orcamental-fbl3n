@echo off
setlocal enabledelayedexpansion

rem ===========================================
rem ETL Production Script - Auto Directory Detection
rem Windows Server 2022 Standard Evaluation
rem ===========================================

rem Auto-detect production directory (where this script is located)
set "PROD_DIR=%~dp0"
if "%PROD_DIR:~-1%"=="\" set "PROD_DIR=%PROD_DIR:~0,-1%"

rem Create subdirectories
set "LOG_DIR=%PROD_DIR%\logs"

echo ===== CONFIGURACAO AUTOMATICA =====
echo Pasta de producao: %PROD_DIR%
echo Pasta de logs: %LOG_DIR%
echo.

rem Create directories if they don't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

rem Navigate to production directory
cd /d "%PROD_DIR%" || (
    echo ERRO: Nao foi possivel acessar o diretorio de producao
    echo %date% %time% - Falha ao acessar diretorio: %PROD_DIR% >> "%LOG_DIR%\deployment_errors.log"
    exit /b 1
)

rem Log start time
echo %date% %time% - Iniciando processo ETL em: %PROD_DIR% >> "%LOG_DIR%\etl_execution.log"

rem Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado no sistema
    echo %date% %time% - Python nao encontrado >> "%LOG_DIR%\deployment_errors.log"
    echo.
    echo Instale Python e adicione ao PATH do sistema
    pause
    exit /b 1
)

rem Check if virtual environment exists
if not exist "venv" (
    echo Criando ambiente virtual de producao...
    python -m venv venv
    if errorlevel 1 (
        echo ERRO: Falha ao criar ambiente virtual
        echo %date% %time% - Falha ao criar venv >> "%LOG_DIR%\deployment_errors.log"
        pause
        exit /b 1
    )
    echo %date% %time% - Ambiente virtual criado em: %PROD_DIR%\venv >> "%LOG_DIR%\etl_execution.log"
)

rem Activate virtual environment
echo Ativando ambiente virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERRO: Falha ao ativar ambiente virtual
    echo %date% %time% - Falha ao ativar venv >> "%LOG_DIR%\deployment_errors.log"
    pause
    exit /b 1
)

rem Install/update dependencies
if exist "requirements.txt" (
    echo Instalando/atualizando dependencias...
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo AVISO: Algumas dependencias podem nao ter sido instaladas
        echo %date% %time% - Problemas na instalacao de dependencias >> "%LOG_DIR%\deployment_errors.log"
    ) else (
        echo %date% %time% - Dependencias instaladas com sucesso >> "%LOG_DIR%\etl_execution.log"
    )
) else (
    echo AVISO: requirements.txt nao encontrado em: %PROD_DIR%
    echo %date% %time% - requirements.txt nao encontrado >> "%LOG_DIR%\deployment_errors.log"
)

rem Check for main.py
if not exist "main.py" (
    echo ERRO: main.py nao encontrado em: %PROD_DIR%
    echo %date% %time% - main.py nao encontrado >> "%LOG_DIR%\deployment_errors.log"
    echo.
    echo Verifique se todos os arquivos foram copiados corretamente
    pause
    exit /b 1
)

rem Execute ETL process
echo.
echo ===== EXECUTANDO PROCESSO ETL =====
echo Iniciando em: %date% %time%
echo Local: %PROD_DIR%
echo.

echo %date% %time% - Iniciando execucao do ETL >> "%LOG_DIR%\etl_execution.log"

python main.py
set "ETL_EXIT_CODE=!errorlevel!"

rem Log completion
echo %date% %time% - ETL finalizado com codigo: !ETL_EXIT_CODE! >> "%LOG_DIR%\etl_execution.log"

rem Deactivate virtual environment
call venv\Scripts\deactivate.bat

rem Show results
echo.
echo ===== RESULTADO FINAL =====
if !ETL_EXIT_CODE! equ 0 (
    echo Status: SUCESSO
    echo ETL executado com sucesso!
    echo Horario: %date% %time%
    echo Logs: %LOG_DIR%\etl_execution.log
    echo %date% %time% - ETL executado com SUCESSO >> "%LOG_DIR%\etl_execution.log"
) else (
    echo Status: ERRO
    echo ETL falhou com codigo: !ETL_EXIT_CODE!
    echo Horario: %date% %time%
    echo Logs de erro: %LOG_DIR%\deployment_errors.log
    echo Logs de execucao: %LOG_DIR%\etl_execution.log
    echo %date% %time% - ETL FALHOU com codigo !ETL_EXIT_CODE! >> "%LOG_DIR%\deployment_errors.log"
)

echo.
echo Pasta de trabalho: %PROD_DIR%
echo ===========================

rem Keep window open if running manually (not from Task Scheduler)
if /i "%1" neq "/scheduled" (
    echo.
    echo Pressione qualquer tecla para sair...
    pause >nul
)

exit /b !ETL_EXIT_CODE!