import React from 'react';
import { 
  Code2, 
  HelpCircle, 
  Award, 
  History, 
  ShieldCheck, 
  User, 
  GraduationCap,
  LogOut,
  Cpu
} from 'lucide-react';
import { CodingSession, AuthUser } from '../types';

export type ActiveTab = 'workspace' | 'viva' | 'assessment' | 'history' | 'instructor';

interface NavbarProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  currentSession?: CodingSession | null;
  serverStatus?: boolean;
  currentUser: AuthUser | null;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  currentUser,
  onLogout,
}) => {
  const isTeacher = currentUser?.role === 'teacher';

  return (
    <header className="sticky top-0 z-50 bg-white border-b border-slate-200 text-slate-800 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div 
          className="flex items-center gap-3 cursor-pointer select-none"
          onClick={() => setActiveTab(isTeacher ? 'instructor' : 'workspace')}
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-600 flex items-center justify-center shadow-md shadow-sky-500/10">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight text-slate-900">DeepMisconception</span>
            </div>
            <p className="text-xs text-slate-500 font-mono">Code Comprehension & Diagnosis</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1">
          {/* Student Tabs: ONLY shown for students */}
          {!isTeacher && (
            <>
              <button
                onClick={() => setActiveTab('workspace')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === 'workspace'
                    ? 'bg-sky-50 text-sky-700 font-bold border border-sky-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Code2 className="w-4 h-4" />
                <span>Workspace</span>
              </button>

              <button
                onClick={() => setActiveTab('history')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === 'history'
                    ? 'bg-sky-50 text-sky-700 font-bold border border-sky-200'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <History className="w-4 h-4" />
                <span>Sessions</span>
              </button>
            </>
          )}

        </nav>

        {/* Right Session & User Profile Controls */}
        <div className="flex items-center gap-3">
          {currentUser && (
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full border text-xs ${
              isTeacher 
                ? 'bg-indigo-50 border-indigo-200 text-indigo-800'
                : 'bg-slate-100 border-slate-200 text-slate-700'
            }`}>
              {isTeacher ? (
                <ShieldCheck className="w-3.5 h-3.5 text-indigo-600" />
              ) : (
                <GraduationCap className="w-3.5 h-3.5 text-sky-600" />
              )}
              <span className="font-semibold text-slate-800">{currentUser.name}</span>
              {currentUser.role === 'student' && currentUser.classroomCode && (
                <span className="px-1.5 py-0.5 rounded bg-sky-100 text-sky-800 font-mono text-[10px] font-bold" title={`Guided by ${currentUser.teacherName || 'Faculty'}`}>
                  {currentUser.classroomCode}
                </span>
              )}
              <span className="text-slate-500 font-mono text-[11px] capitalize">({currentUser.role})</span>
            </div>
          )}

          {/* Switch Role / Logout Button */}
          <button
            onClick={onLogout}
            className="p-1.5 px-2.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors text-xs flex items-center gap-1 font-medium"
            title="Switch user or log out"
          >
            <LogOut className="w-3.5 h-3.5 text-slate-500" />
            <span>Logout</span>
          </button>
        </div>
      </div>
    </header>
  );
};
