
import React, { useState, useEffect } from 'react';
import { Navbar, ActiveTab } from './components/Navbar';
import { LoginView } from './views/LoginView';
import { WorkspaceView } from './views/WorkspaceView';
import { VivaView } from './views/VivaView';
import { AssessmentResultsView } from './views/AssessmentResultsView';
import { SessionHistoryView } from './views/SessionHistoryView';
import { InstructorDashboardView } from './views/InstructorDashboardView';
import { CodingSession, CodeSubmission, AuthUser } from './types';
import { api } from './services/api';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Unhandled UI Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="max-w-xl mx-auto my-12 p-6 bg-white border border-rose-200 rounded-2xl shadow-sm text-center space-y-4">
          <div className="w-12 h-12 mx-auto rounded-full bg-rose-50 border border-rose-200 flex items-center justify-center text-rose-600 font-bold text-xl">
            ⚠️
          </div>
          <h3 className="text-lg font-bold text-slate-900">Something went wrong</h3>
          <p className="text-xs text-slate-600 font-mono bg-slate-50 p-3 rounded-xl border border-slate-200 break-words text-left">
            {this.state.error?.message || 'Unknown render error'}
          </p>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: null });
              if (this.props.onReset) this.props.onReset();
            }}
            className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs rounded-xl shadow-sm transition-colors"
          >
            Reload View
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export const App: React.FC = () => {
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>('workspace');
  const [currentSession, setCurrentSession] = useState<CodingSession | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean>(true);

  // Restore authenticated user from localStorage on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('deepmisconception_auth_user');
    if (savedUser) {
      try {
        const user = JSON.parse(savedUser) as AuthUser;
        setCurrentUser(user);
        if (user.role === 'teacher') {
          setActiveTab('instructor');
        } else {
          setActiveTab('workspace');
        }
      } catch {
        // ignore corrupted user record
      }
    }
  }, []);

  // Poll backend health
  useEffect(() => {
    const checkServer = async () => {
      try {
        const health = await api.getHealth();
        if (health.status === 'healthy') {
          setServerOnline(true);
        }
      } catch (err) {
        setServerOnline(false);
      }
    };
    checkServer();
    const interval = setInterval(checkServer, 15000);
    return () => clearInterval(interval);
  }, []);

  // Restore current session from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('deepmisconception_active_session');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setCurrentSession(parsed);
      } catch {
        // ignore corrupted session record
      }
    }
  }, []);

  // Ensure an active session is auto-created whenever a student is logged in without one
  useEffect(() => {
    if (currentUser?.role === 'student' && !currentSession) {
      api.createSession({
        student_id: currentUser.id,
        student_name: currentUser.name,
        classroom_id: currentUser.classroomId,
        classroom_name: currentUser.classroomName,
        teacher_id: currentUser.teacherId,
        teacher_name: currentUser.teacherName,
        title: 'Code Comprehension Assessment',
      }).then((sess) => {
        setCurrentSession(sess);
        localStorage.setItem('deepmisconception_active_session', JSON.stringify(sess));
      }).catch((e) => {
        console.error('Failed auto-creating student session:', e);
      });
    }
  }, [currentUser, currentSession]);

  const handleLogin = async (user: AuthUser) => {
    setCurrentUser(user);
    localStorage.setItem('deepmisconception_auth_user', JSON.stringify(user));
    if (user.role === 'teacher') {
      setActiveTab('instructor');
    } else {
      let session = currentSession;
      if (!session || session.student_id !== user.id) {
        try {
          session = await api.createSession({
            student_id: user.id,
            student_name: user.name,
            classroom_id: user.classroomId,
            classroom_name: user.classroomName,
            teacher_id: user.teacherId,
            teacher_name: user.teacherName,
            title: 'Code Comprehension Assessment',
          });
          setCurrentSession(session);
          localStorage.setItem('deepmisconception_active_session', JSON.stringify(session));
        } catch (err) {
          console.error('Failed to create session on login', err);
        }
      }
      setActiveTab('workspace');
    }
  };

  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem('deepmisconception_auth_user');
    setActiveTab('workspace');
  };

  const handleSelectSession = (session: CodingSession) => {
    setCurrentSession(session);
    localStorage.setItem('deepmisconception_active_session', JSON.stringify(session));
    setActiveTab('workspace');
  };

  const handleCodeSubmitted = (submission: CodeSubmission) => {
    // Code submitted
  };

  // Safe tab change enforcing role boundary
  const handleTabChange = (targetTab: ActiveTab) => {
    if (targetTab === 'instructor' && currentUser?.role !== 'teacher') {
      // Students cannot enter instructor studio
      setActiveTab('workspace');
      return;
    }
    setActiveTab(targetTab);
  };

  // If not logged in, show the Login Page
  if (!currentUser) {
    return <LoginView onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans selection:bg-sky-500/20 selection:text-sky-900">
      <Navbar
        activeTab={activeTab}
        setActiveTab={handleTabChange}
        currentSession={currentSession}
        serverStatus={serverOnline}
        currentUser={currentUser}
        onLogout={handleLogout}
      />

      <main className="flex-1 pb-12">
        <ErrorBoundary onReset={() => setActiveTab('workspace')}>
          {activeTab === 'workspace' && (
            <WorkspaceView
              session={currentSession}
              currentUser={currentUser}
              onCodeSubmitted={handleCodeSubmitted}
              onNavigateToViva={() => setActiveTab('viva')}
            />
          )}

          {activeTab === 'viva' && (
            <VivaView
              session={currentSession}
              onNavigateToResults={() => setActiveTab('assessment')}
            />
          )}

          {activeTab === 'assessment' && (
            <AssessmentResultsView
              session={currentSession}
              currentUser={currentUser}
              onNavigateToWorkspace={() => setActiveTab('workspace')}
              onSelectSession={handleSelectSession}
            />
          )}

          {activeTab === 'history' && (
            <SessionHistoryView
              onSelectSession={handleSelectSession}
              onViewReport={(session) => {
                handleSelectSession(session);
                setActiveTab('assessment');
              }}
              currentSessionId={currentSession?.id}
              currentUser={currentUser}
            />
          )}

          {/* Instructor Studio: Strictly rendered for teacher role only */}
          {activeTab === 'instructor' && currentUser.role === 'teacher' && (
            <InstructorDashboardView
              onSelectSession={handleSelectSession}
              currentUser={currentUser}
            />
          )}
        </ErrorBoundary>
      </main>

      <footer className="py-4 border-t border-slate-200 bg-white text-center text-xs text-slate-500 font-mono flex items-center justify-center gap-4">
        <span>DeepMisconception &bull; AI-Assisted Code Comprehension Verification</span>
        <span className="text-slate-300">&bull;</span>
        <span className="text-slate-600 capitalize font-medium">Logged in as {currentUser.role}: {currentUser.name}</span>
      </footer>
    </div>
  );
};

export default App;
