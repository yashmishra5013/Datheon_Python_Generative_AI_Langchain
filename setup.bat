@echo off
echo === Creating virtual environment ===
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
echo === Installing requirements (this can take a few minutes) ===
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo DONE! Now open the .env file and paste your API key (GROQ_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY).
pause
