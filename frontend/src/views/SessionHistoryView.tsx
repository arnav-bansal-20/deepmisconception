import React, { useState, useEffect } from 'react';
import { 
  History, 
  Search, 
  User, 
  FileCode, 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  ArrowRight,
  RefreshCw,
  Trash2
} from 'lucide-react';
import { CodingSession, AuthUser } from '../types';
import { api } from '../services/api';

interface SessionHistoryViewProps {
  onSelectSession: (session: CodingSession) => void;
  currentSessionId?: string;
  currentUser?: AuthUser | null;
}

export const SessionHistoryView: React.FC<SessionHistoryViewProps> = ({
  onSelectSession,
  currentSessionId,
  currentUser,
}) => {
  const [sessions, setSessions] = useState<CodingSession[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const fetchSessions = async () => {
    setIsLoading(true);
    try {
      const isStudent = currentUser?.role === 'student';
      const params = isStudent && currentUser?.id ? { student_id: currentUser.id } : undefined;
      const data = await api.listSessions(params);

      // Enforce strict student boundary: students only see their own sessions
      const filtered = isStudent && currentUser?.id
        ? data.filter((s) => s.student_id === currentUser.id)
        : data;

      setSessions(filtered);
    } catch (err) {
      console.error('Failed to fetch sessions', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, [currentUser?.id, currentUser?.role]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm('Delete this session and all associated data?')) return;
    try {
      await api.deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
    } catch (err) {
      console.error('Failed to delete session', err);
    }
  };

  const filteredSessions = sessions.filter(
    (s) =>
      s.student_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.student_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="max-w-6xl mx-auto space-y-6 px-4 sm:px-6 py-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-slate-200 p-4 rounded-2xl shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200 font-bold uppercase">
              Audit & Archive
            </span>
            <h2 className="text-lg font-bold text-slate-900">
              {currentUser?.role === 'student' ? 'My Coding Sessions' : 'Student Session History'}
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            {currentUser?.role === 'student'
              ? 'Browse, resume, or inspect your own previous coding sessions and evaluations.'
              : 'Browse, resume, or inspect previous coding sessions and their evaluation verdicts.'}
          </p>
        </div>

        <button
          onClick={fetchSessions}
          disabled={isLoading}
          className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors text-xs font-medium flex items-center gap-1.5"
          title="Refresh list"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Search Input */}
      <div className="relative">
        <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search by student name, ID, or task title..."
          className="w-full bg-white border border-slate-300 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 shadow-sm"
        />
      </div>

      {/* Sessions Table / Cards */}
      {filteredSessions.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-2xl border border-slate-200 p-8 text-slate-500 text-xs shadow-sm">
          {isLoading ? 'Loading session archives...' : 'No coding sessions recorded yet.'}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredSessions.map((s) => {
            const isCurrent = s.id === currentSessionId;
            return (
              <div
                key={s.id}
                onClick={() => onSelectSession(s)}
                className={`p-5 rounded-2xl cursor-pointer transition-all border text-left flex flex-col justify-between ${
                  isCurrent
                    ? 'bg-sky-50/60 border-sky-400 ring-1 ring-sky-400 shadow-sm'
                    : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/60 shadow-sm'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono font-bold text-sky-700 uppercase">
                      {s.language}
                    </span>
                    <div className="flex items-center gap-2">
                      {s.latest_verdict && (
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                          s.latest_verdict === 'Understood'
                            ? 'bg-emerald-50 text-emerald-800 border border-emerald-300'
                            : s.latest_verdict === 'Partial Understanding'
                            ? 'bg-amber-50 text-amber-800 border border-amber-300'
                            : 'bg-rose-50 text-rose-800 border border-rose-300'
                        }`}>
                          {s.latest_verdict}
                        </span>
                      )}
                      <button
                        onClick={(e) => handleDelete(e, s.id)}
                        className="text-slate-400 hover:text-rose-600 p-1"
                        title="Delete session"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <h3 className="font-bold text-slate-900 text-sm mb-1">{s.title}</h3>
                  <div className="flex items-center gap-1.5 text-xs text-slate-600 mb-3">
                    <User className="w-3.5 h-3.5 text-slate-400" />
                    <span>{s.student_name}</span>
                    <span className="text-slate-500 font-mono text-[11px]">({s.student_id})</span>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500 font-mono">
                  <span>{new Date(s.created_at).toLocaleDateString()}</span>
                  <div className="flex items-center gap-1 text-sky-700 font-bold group">
                    <span>{isCurrent ? 'Active Now' : 'Open Session'}</span>
                    <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
