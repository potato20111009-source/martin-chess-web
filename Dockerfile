# 步驟 1：使用 .NET SDK 編譯 C# 後端
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY . .
RUN dotnet publish -c Release -o /app/publish

# 步驟 2：建立包含 .NET 8 與 Python 3 + ONNX 的 Linux 執行環境
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

# 安裝 Python 3 及 AI 模型推論套件 (ONNX Runtime / NumPy / Chess)
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir onnxruntime numpy python-chess --break-system-packages && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /app/publish .
# 強制複製 engines 資料夾至 Linux 執行目錄
COPY engines ./engines

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "martin-chess-web.dll"]
