# 步驟 1：使用 .NET SDK 編譯 C# 後端
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY . .
RUN dotnet publish -c Release -o /app/publish

# 步驟 2：建立包含 .NET 8 與 Python 3 + PyTorch + ONNX 的 Linux 執行環境
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# 安裝 Python 3 基礎套件
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# 使用 --extra-index-url 正確安裝 CPU 版 PyTorch
RUN pip3 install --no-cache-dir --break-system-packages torch --extra-index-url https://download.pytorch.org/whl/cpu
# 安裝 ONNX、NumPy、python-chess 與 tqdm
RUN pip3 install --no-cache-dir --break-system-packages onnxruntime numpy python-chess tqdm

COPY --from=build /app/publish .
# 強制複製 engines 資料夾至 Linux 執行目錄
COPY engines ./engines

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "martin-chess-web.dll"]