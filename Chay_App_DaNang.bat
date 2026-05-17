@echo off
title Khoi dong App Du bao Giao thong Da Nang
color 0B

echo ======================================================
echo    DANG KHOI DONG UNG DUNG DU BAO HOMITUTOR (AI)
echo ======================================================
echo.

:: 1. Chuyen vao o dia D va thu muc do an
cd /d D:\CDIOFN

:: 2. Kiem tra xem moi truong ao co ton tai khong
if not exist ".venv\Scripts\activate" (
    color 0C
    echo [LOI] Khong tim thay moi truong ao tai D:\CDIOFN\.venv
    echo Vui long kiem tra lai duong dan!
    pause
    exit
)

:: 3. Chạy ứng dụng Streamlit
echo Dang kich hoat moi truong ao va chay Dashboard...
echo.
".venv\Scripts\streamlit" run app_danang.py

echo.
echo ======================================================
echo    UNG DUNG DA DONG. NHAN PHIM BAT KY DE THOAT.
echo ======================================================
pause