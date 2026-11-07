using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

[ApiController]
[Route("api")]
public sealed class EngineController : ControllerBase
{
    private static readonly TimeSpan EngineTimeout = TimeSpan.FromMilliseconds(12000);
    private readonly IWebHostEnvironment _environment;
    private readonly ILogger<EngineController> _logger;

    public EngineController(
        IWebHostEnvironment environment,
        ILogger<EngineController> logger)
    {
        _environment = environment;
        _logger = logger;
    }

    [HttpPost("move")]
    public async Task<IActionResult> GetMove([FromBody] MoveRequest request)
    {
        if (request == null ||
            string.IsNullOrWhiteSpace(request.Fen) ||
            request.Fen.Length > 256 ||
            request.Fen.Any(char.IsControl))
        {
            return BadRequest(new { error = "A valid FEN string is required." });
        }

        WriteDiagnostic($"Received FEN: {request.Fen}");
        var simulations = request.Simulations < 50
            ? 100
            : request.Simulations > 1000
                ? 400
                : request.Simulations;

        var executablePath = ResolveEnginePath();
        if (executablePath == null)
        {
            _logger.LogError("MartinAI executable was not found under {ContentRoot}.", _environment.ContentRootPath);
            return StatusCode(500, new { error = "The chess engine executable was not found." });
        }

        using var timeout = new CancellationTokenSource(EngineTimeout);
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = executablePath,
                WorkingDirectory = Path.GetDirectoryName(executablePath)!,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            }
        };

        var processStarted = false;
        try
        {
            if (!process.Start())
            {
                throw new InvalidOperationException("The chess engine process could not be started.");
            }
            processStarted = true;

            var standardErrorTask = process.StandardError.ReadToEndAsync();
            await process.StandardInput.WriteLineAsync("uci");
            await process.StandardInput.FlushAsync();
            await ReadUntilAsync(process, "uciok", timeout.Token);

            await process.StandardInput.WriteLineAsync("isready");
            await process.StandardInput.FlushAsync();
            await ReadUntilAsync(process, "readyok", timeout.Token);

            await process.StandardInput.WriteAsync($"position fen {request.Fen}\n");
            await process.StandardInput.WriteAsync($"go nodes {simulations}\n");
            await process.StandardInput.FlushAsync();

            var bestMoveLine = await ReadUntilAsync(process, "bestmove ", timeout.Token);
            WriteDiagnostic($"MartinAI raw bestmove output: {bestMoveLine}");
            var bestMove = bestMoveLine.Substring("bestmove ".Length)
                .Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .FirstOrDefault();

            if (string.IsNullOrEmpty(bestMove))
            {
                throw new InvalidOperationException("The chess engine returned an invalid best move.");
            }

            await process.StandardInput.WriteLineAsync("quit");
            await process.StandardInput.FlushAsync();
            await process.WaitForExitAsync(timeout.Token);
            await standardErrorTask;

            return Ok(new { bestMove });
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested)
        {
            _logger.LogWarning("MartinAI did not respond within {TimeoutMs} ms.", EngineTimeout.TotalMilliseconds);
            return StatusCode(408, new { error = "timeout", bestMove = (string?)null });
        }
        catch (TimeoutException exception)
        {
            _logger.LogWarning(exception, "MartinAI timed out while returning a move.");
            return StatusCode(408, new { error = "timeout", bestMove = (string?)null });
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Failed to get a move from MartinAI.");
            return StatusCode(500, new { error = "Failed to get a move from the chess engine." });
        }
        finally
        {
            if (processStarted && !process.HasExited)
            {
                process.Kill(true);
            }
        }
    }

    private static async Task<string> ReadUntilAsync(
        Process process,
        string expectedLine,
        CancellationToken cancellationToken)
    {
        while (true)
        {
            var line = await process.StandardOutput.ReadLineAsync().WaitAsync(cancellationToken);
            if (line == null)
            {
                throw new InvalidOperationException(
                    $"The chess engine exited before returning '{expectedLine}'.");
            }

            if (expectedLine.EndsWith(" ", StringComparison.Ordinal)
                ? line.StartsWith(expectedLine, StringComparison.Ordinal)
                : string.Equals(line, expectedLine, StringComparison.Ordinal))
            {
                return line;
            }
        }
    }

    private static void WriteDiagnostic(string message)
    {
        Console.WriteLine(message);
        Debug.WriteLine(message);
    }

    private string? ResolveEnginePath()
    {
        var paths = new[]
        {
            Path.Combine(_environment.ContentRootPath, "Engine", "MartinAI", "MartinAI.exe"),
            Path.Combine(_environment.ContentRootPath, "engines", "MartinAI", "MartinAI.exe")
        };

        return paths.FirstOrDefault(System.IO.File.Exists);
    }
}

public sealed class MoveRequest
{
    public string? Fen { get; set; }
    public int Simulations { get; set; } = 200;
}
