import React from 'react';
import { Terminal, Shield, CheckCircle, AlertTriangle, Clock } from 'lucide-react';

interface ExecutionTerminalProps {
  stdout: string | null;
  stderr: string | null;
  isRunning: boolean;
  executionTimeMs?: number | null;
  sandboxUsed?: boolean;
}

export const ExecutionTerminal: React.FC<ExecutionTerminalProps> = ({
  stdout,
  stderr,
  isRunning,
  executionTimeMs,
  sandboxUsed = true,
}) => {
  return (
    <div className="flex flex-col h-full bg-slate-50 rounded-xl border border-slate-300 overflow-hidden text-slate-800 shadow-sm">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-white border-b border-slate-200 text-xs select-none">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-sky-600" />
          <span className="font-bold text-slate-800">Execution Output</span>
          {sandboxUsed && (
            <span className="flex items-center gap-1 text-[10px] bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full font-mono font-semibold">
              <Shield className="w-2.5 h-2.5" />
              Isolated Sandbox
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 text-slate-500 font-mono">
          {isRunning ? (
            <span className="flex items-center gap-1.5 text-amber-600 font-bold">
              <span className="w-2 h-2 rounded-full bg-amber-500 animate-ping" />
              Executing...
            </span>
          ) : (
            executionTimeMs !== undefined && executionTimeMs !== null && (
              <span className="flex items-center gap-1 text-slate-600">
                <Clock className="w-3 h-3 text-slate-400" />
                {executionTimeMs.toFixed(1)} ms
              </span>
            )
          )}
        </div>
      </div>

      {/* Terminal Body */}
      <div className="flex-1 p-4 font-mono text-xs overflow-y-auto space-y-2 select-text bg-white">
        {isRunning && (
          <div className="text-amber-700 animate-pulse font-semibold">
            Executing code in isolated subprocess environment...
          </div>
        )}

        {!isRunning && !stdout && !stderr && (
          <div className="text-slate-400 italic">
            Click "Run Code" to execute in the secure sandbox environment. Output and runtime diagnostic traces will appear here.
          </div>
        )}

        {stdout && (
          <div>
            <div className="text-slate-600 text-[11px] font-bold uppercase tracking-wider mb-1 flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-emerald-600" />
              Standard Output:
            </div>
            <pre className="text-slate-900 bg-emerald-50/60 p-2.5 rounded-lg border border-emerald-200 whitespace-pre-wrap">
              {stdout}
            </pre>
          </div>
        )}

        {stderr && (
          <div>
            <div className="text-rose-700 text-[11px] font-bold uppercase tracking-wider mb-1 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3 text-rose-600" />
              Standard Error / Traceback:
            </div>
            <pre className="text-rose-900 bg-rose-50 p-2.5 rounded-lg border border-rose-200 whitespace-pre-wrap">
              {stderr}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
