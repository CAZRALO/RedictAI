FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và các thư viện khác
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Mở cổng 5150
EXPOSE 5150

# Chạy ứng dụng bằng gunicorn (bảo mật và hiệu năng tốt hơn cho production)
CMD ["gunicorn", "-b", "0.0.0.0:5150", "--timeout", "120", "app:app"]
