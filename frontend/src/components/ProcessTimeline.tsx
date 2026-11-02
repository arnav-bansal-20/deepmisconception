import React, { useState } from 'react';
import { 
  Activity, 
  Clock, 
  Clipboard, 
  Play, 
  AlertTriangle, 
  CheckCircle2, 
  ShieldAlert, 
  Zap, 
  HelpCircle,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { ProcessFeatures, TimelineMilestone } from '../types';

interface ProcessTimelineProps {
  features: ProcessFeatures | null;
  isLoading?: boolean;
}

export const ProcessTimeline: React.FC<ProcessTimelineProps> = ({
  features,
  isLoading = false,
}) => {
  const [selectedMilestone, setSelectedMilestone] = useState<TimelineMilestone | null>(null);

  if (isLoading) {
    return (
      <div className="p-6 bg-white border border-slate-200 rounded-2xl animate-pulse space-y-4">
        <div className="h-4 bg-slate-200 rounded w-1/3"></div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-16 bg-slate-100 rounded-xl"></div>
          ))}
        </div>
      </div>
    );
  }

  if (!features || features.total_events === 0) {
    return (
      <div className="p-6 bg-white border border-slate-200 rounded-2xl text-center space-y-2">
        <Activity className="w-8 h-8 text-slate-300 mx-auto" />
        <p className="text-xs font-semibold text-slate-600">No coding telemetry collected yet</p>
        <p className="text-[11px] text-slate-400">
          Telemetry captures keystroke rhythms, editing patterns, and execution runs as you work in the editor.
        </p>
      </div>
    );
  }

  const burstPct = Math.round((features.burstiness_ratio || 0) * 100);
  const recoveryPct = Math.round((features.error_recovery_rate ?? 1.0) * 100);
  const maxDuration = Math.max(1, features.coding_duration_seconds);

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-5">
      {/* Header and Pedagogical Profile */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-sky-600" />
          <h3 className="font-bold text-slate-900 text-sm">Coding-Process Telemetry & Timeline</h3>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono uppercase text-slate-500 font-bold">Profile:</span>
          <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${
            features.process_profile?.includes('Block')
              ? 'bg-amber-50 text-amber-800 border-amber-300'
              : features.process_profile?.includes('Debugger')
              ? 'bg-indigo-50 text-indigo-800 border-indigo-300'
              : features.process_profile?.includes('Continuous')
              ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
              : 'bg-sky-50 text-sky-800 border-sky-300'
          }`}>
            {features.process_profile || 'Balanced Incremental Developer'}
          </span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
          <div className="text-[10px] font-mono uppercase text-slate-500 flex items-center gap-1 mb-1 font-bold">
            <Clipboard className="w-3 h-3 text-sky-600" />
            <span>Editing Activity</span>
          </div>
          <div className="text-lg font-bold font-mono text-slate-900">
            {features.edit_count} <span className="text-xs font-normal text-slate-500">edits</span>
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">{features.total_events} total events recorded</div>
        </div>

        <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
          <div className="text-[10px] font-mono uppercase text-slate-500 flex items-center gap-1 mb-1 font-bold">
            <Zap className="w-3 h-3 text-sky-600" />
            <span>Burstiness</span>
          </div>
          <div className="text-lg font-bold font-mono text-slate-900">
            {burstPct}%
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">{features.burst_count} fast edits (&lt;0.5s)</div>
        </div>

        <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
          <div className="text-[10px] font-mono uppercase text-slate-500 flex items-center gap-1 mb-1 font-bold">
            <Clock className="w-3 h-3 text-indigo-600" />
            <span>Total Duration</span>
          </div>
          <div className="text-lg font-bold font-mono text-slate-900">
            {features.coding_duration_seconds.toFixed(0)}s
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">Avg interval {features.average_edit_interval_seconds}s</div>
        </div>

        <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
          <div className="text-[10px] font-mono uppercase text-slate-500 flex items-center gap-1 mb-1 font-bold">
            <Play className="w-3 h-3 text-emerald-600" />
            <span>Sandbox Runs</span>
          </div>
          <div className="text-lg font-bold font-mono text-slate-900">
            {features.run_count}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            1st run at {features.time_to_first_run_seconds != null ? `${features.time_to_first_run_seconds}s` : 'N/A'}
          </div>
        </div>

        <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
          <div className="text-[10px] font-mono uppercase text-slate-500 flex items-center gap-1 mb-1 font-bold">
            <TrendingUp className="w-3 h-3 text-teal-600" />
            <span>Error Recovery</span>
          </div>
          <div className={`text-lg font-bold font-mono ${
            recoveryPct >= 80 ? 'text-emerald-700' : 'text-amber-600'
          }`}>
            {recoveryPct}%
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">{features.error_count || 0} run error(s)</div>
        </div>

        <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
          <div className="text-[10px] font-mono uppercase text-slate-500 flex items-center gap-1 mb-1 font-bold">
            <Cpu className="w-3 h-3 text-indigo-600" />
            <span>Typing Pace</span>
          </div>
          <div className="text-lg font-bold font-mono text-slate-900">
            {features.typing_speed_cpm || 0}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">Chars / minute</div>
        </div>
      </div>

      {/* Interactive Chronological Visual Timeline Track */}
      {features.timeline_events && features.timeline_events.length > 0 && (
        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-bold text-slate-700 flex items-center gap-1.5">
              <span>Timeline Chronology</span>
              <span className="text-[10px] font-normal text-slate-400">(click an event node to inspect)</span>
            </span>
            <span className="text-slate-500 text-[11px]">0s &rarr; {features.coding_duration_seconds.toFixed(0)}s</span>
          </div>

          {/* Visual Track Bar */}
          <div className="relative h-6 bg-slate-100 rounded-full border border-slate-200 flex items-center px-2">
            <div className="absolute inset-x-2 h-1 bg-slate-200 rounded-full" />
            {features.timeline_events.map((m, idx) => {
              const posPercent = Math.min(100, Math.max(0, (m.offset_seconds / maxDuration) * 100));
              const isSelected = selectedMilestone === m;
              return (
                <button
                  key={idx}
                  onClick={() => setSelectedMilestone(m)}
                  style={{ left: `${posPercent}%` }}
                  title={`${m.label} at +${m.offset_seconds}s`}
                  className={`absolute -translate-x-1/2 w-4 h-4 rounded-full border-2 transition-all transform hover:scale-125 focus:outline-none ${
                    isSelected ? 'ring-2 ring-indigo-500 scale-125 z-10' : ''
                  } ${
                    m.event_type === 'paste'
                      ? 'bg-amber-400 border-white'
                      : m.event_type === 'error'
                      ? 'bg-rose-500 border-white'
                      : m.event_type === 'run'
                      ? 'bg-emerald-500 border-white'
                      : 'bg-indigo-600 border-white'
                  }`}
                />
              );
            })}
          </div>

          {/* Selected Node Details or Recent List */}
          {selectedMilestone ? (
            <div className="p-3 bg-indigo-50/70 border border-indigo-200 rounded-xl text-xs space-y-1 animate-fadeIn">
              <div className="flex items-center justify-between">
                <span className="font-bold text-indigo-900 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-indigo-600" />
                  <span>+{selectedMilestone.offset_seconds}s: {selectedMilestone.label}</span>
                </span>
                <button
                  onClick={() => setSelectedMilestone(null)}
                  className="text-slate-400 hover:text-slate-600 font-bold"
                >
                  &times;
                </button>
              </div>
              <p className="text-slate-700 font-mono text-[11px]">{selectedMilestone.detail}</p>
            </div>
          ) : (
            <div className="flex flex-wrap gap-2 pt-1">
              {features.timeline_events.slice(0, 4).map((m, idx) => (
                <div key={idx} className="px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-[10px] font-mono text-slate-600 flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${
                    m.event_type === 'paste' ? 'bg-amber-500' : m.event_type === 'error' ? 'bg-rose-500' : 'bg-emerald-500'
                  }`} />
                  <span>+{m.offset_seconds}s: {m.label}</span>
                </div>
              ))}
              {features.timeline_events.length > 4 && (
                <span className="text-[10px] font-mono text-slate-400 self-center">
                  +{features.timeline_events.length - 4} more events recorded
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Ethical Academic Safeguard Notice */}
      <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-[11px] text-slate-600 flex items-start gap-2">
        <ShieldAlert className="w-4 h-4 text-sky-600 flex-shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <span className="font-bold text-slate-800">Ethical Telemetry Guarantee:</span>
          <p className="leading-relaxed">
            {features.ethical_disclaimer}
          </p>
        </div>
      </div>
    </div>
  );
};
