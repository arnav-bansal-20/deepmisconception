import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  GraduationCap, 
  ShieldCheck, 
  ArrowRight, 
  User, 
  School,
  CheckCircle2,
  Users,
  BookOpen,
  Hash
} from 'lucide-react';
import { AuthUser, UserRole, Classroom } from '../types';
import { api } from '../services/api';

interface LoginViewProps {
  onLogin: (user: AuthUser) => void;
}

export const LoginView: React.FC<LoginViewProps> = ({ onLogin }) => {
  const [role, setRole] = useState<UserRole>('student');
  const [classrooms, setClassrooms] = useState<Classroom[]>([]);
  const [selectedClassroomId, setSelectedClassroomId] = useState<string>('class-alpha-101');
  
  // Custom class code entry mode
  const [useCustomCode, setUseCustomCode] = useState(false);
  const [customCode, setCustomCode] = useState('');
  const [customCodeError, setCustomCodeError] = useState<string | null>(null);

  // Student form fields
  const [studentId, setStudentId] = useState('');
  const [studentName, setStudentName] = useState('');
  
  // Teacher form fields
  const [teacherId, setTeacherId] = useState('');
  const [teacherName, setTeacherName] = useState('');
  const [department, setDepartment] = useState('');

  // Load classrooms on mount
  useEffect(() => {
    const fetchClassrooms = async () => {
      try {
        const list = await api.listClassrooms();
        setClassrooms(list);
        if (list.length > 0 && !selectedClassroomId) {
          setSelectedClassroomId(list[0].id);
        }
      } catch (err) {
        console.error('Failed to load classrooms', err);
      }
    };
    fetchClassrooms();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (role === 'student') {
      if (!studentId.trim() || !studentName.trim()) return;

      let targetClassroom: Classroom | undefined;
      if (useCustomCode) {
        if (!customCode.trim()) {
          setCustomCodeError('Please enter a valid class code');
          return;
        }
        try {
          targetClassroom = await api.getClassroomByCode(customCode.trim());
        } catch {
          setCustomCodeError(`Classroom code "${customCode}" not found.`);
          return;
        }
      } else {
        targetClassroom = classrooms.find((c) => c.id === selectedClassroomId);
      }

      onLogin({
        role: 'student',
        id: studentId.trim(),
        name: studentName.trim(),
        classroomId: targetClassroom?.id || 'class-alpha-101',
        classroomName: targetClassroom?.name || 'CS101: Python Foundations - Group Alpha',
        classroomCode: targetClassroom?.code || 'CS101-ALPHA',
        teacherId: targetClassroom?.teacher_id || 'FAC-CS-804',
        teacherName: targetClassroom?.teacher_name || 'Dr. Ada Smith',
      });
    } else {
      if (!teacherId.trim() || !teacherName.trim()) return;
      onLogin({
        role: 'teacher',
        id: teacherId.trim(),
        name: teacherName.trim(),
        department: department.trim(),
      });
    }
  };

  const selectedClass = classrooms.find((c) => c.id === selectedClassroomId);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col justify-center items-center p-4 sm:p-6 relative overflow-hidden">
      {/* Background ambient lighting */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-gradient-to-tr from-sky-400/10 via-indigo-400/10 to-transparent blur-3xl pointer-events-none rounded-full" />

      <div className="max-w-md w-full space-y-6 relative z-10">
        {/* Header Logo */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-tr from-sky-600 to-indigo-600 shadow-md shadow-sky-500/10 mb-1">
            <Cpu className="w-8 h-8 text-white" />
          </div>
          <div className="flex items-center justify-center gap-2">
            <h1 className="text-2xl font-black text-slate-900 tracking-tight">DeepMisconception</h1>
          </div>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            AI-Assisted Code Comprehension Verification & Student Misconception Detection
          </p>
        </div>

        {/* Login Container Card */}
        <div className="bg-white border border-slate-200 rounded-3xl p-6 sm:p-8 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-sky-500 via-indigo-500 to-cyan-500" />

          {/* Role Selection Switcher */}
          <div className="space-y-2 mb-6">
            <label className="block text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-500 text-center">
              Select Your Academic Role
            </label>
            <div className="grid grid-cols-2 gap-3 p-1 bg-slate-100 rounded-2xl border border-slate-200">
              <button
                type="button"
                onClick={() => setRole('student')}
                className={`flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-bold transition-all ${
                  role === 'student'
                    ? 'bg-sky-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                }`}
              >
                <GraduationCap className="w-4 h-4" />
                <span>Student</span>
              </button>

              <button
                type="button"
                onClick={() => setRole('teacher')}
                className={`flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-bold transition-all ${
                  role === 'teacher'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                }`}
              >
                <ShieldCheck className="w-4 h-4" />
                <span>Teacher / Faculty</span>
              </button>
            </div>
          </div>

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {role === 'student' ? (
              <>
                {/* Classroom / Teacher Group Selector */}
                <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-2xl space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-[11px] font-mono font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5 text-sky-600" />
                      <span>Guiding Teacher & Group</span>
                    </label>
                    <button
                      type="button"
                      onClick={() => {
                        setUseCustomCode(!useCustomCode);
                        setCustomCodeError(null);
                      }}
                      className="text-[10px] text-sky-600 hover:text-sky-800 font-semibold"
                    >
                      {useCustomCode ? 'Choose from list' : 'Have a class code?'}
                    </button>
                  </div>

                  {!useCustomCode ? (
                    <div className="space-y-1">
                      <select
                        value={selectedClassroomId}
                        onChange={(e) => setSelectedClassroomId(e.target.value)}
                        className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500"
                      >
                        {classrooms.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.teacher_name} — {c.name} ({c.code})
                          </option>
                        ))}
                      </select>
                      {selectedClass && (
                        <p className="text-[11px] text-slate-500 px-1 pt-1 flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                          <span>Code: <strong className="font-mono text-slate-700">{selectedClass.code}</strong> • Guided by <strong className="text-slate-700">{selectedClass.teacher_name}</strong></span>
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-1">
                      <div className="relative">
                        <Hash className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                        <input
                          type="text"
                          value={customCode}
                          onChange={(e) => {
                            setCustomCode(e.target.value);
                            setCustomCodeError(null);
                          }}
                          placeholder="Enter Group Code"
                          className="w-full uppercase font-mono bg-white border border-slate-300 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500"
                        />
                      </div>
                      {customCodeError && (
                        <p className="text-[11px] text-rose-600 px-1">{customCodeError}</p>
                      )}
                    </div>
                  )}
                </div>

                <div>
                  <label className="block text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                    Student ID / Roll Number
                  </label>
                  <div className="relative">
                    <User className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      required
                      value={studentId}
                      onChange={(e) => setStudentId(e.target.value)}
                      onInvalid={(e) => (e.target as HTMLInputElement).setCustomValidity('Enter Student ID')}
                      onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                      placeholder="Enter Student ID"
                      className="w-full bg-slate-50 border border-slate-300 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:bg-white"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                    Full Name
                  </label>
                  <div className="relative">
                    <BookOpen className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      required
                      value={studentName}
                      onChange={(e) => setStudentName(e.target.value)}
                      onInvalid={(e) => (e.target as HTMLInputElement).setCustomValidity('Enter Full Name')}
                      onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                      placeholder="Enter Full Name"
                      className="w-full bg-slate-50 border border-slate-300 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:bg-white"
                    />
                  </div>
                </div>
              </>
            ) : (
              <>
                <div>
                  <label className="block text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                    Faculty / Teacher ID
                  </label>
                  <div className="relative">
                    <School className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      required
                      value={teacherId}
                      onChange={(e) => setTeacherId(e.target.value)}
                      onInvalid={(e) => (e.target as HTMLInputElement).setCustomValidity('Enter Faculty ID')}
                      onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                      placeholder="Enter Faculty ID"
                      className="w-full bg-slate-50 border border-slate-300 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                    Faculty Full Name
                  </label>
                  <div className="relative">
                    <User className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      required
                      value={teacherName}
                      onChange={(e) => setTeacherName(e.target.value)}
                      onInvalid={(e) => (e.target as HTMLInputElement).setCustomValidity('Enter Faculty Name')}
                      onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                      placeholder="Enter Faculty Name"
                      className="w-full bg-slate-50 border border-slate-300 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-[11px] font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                    Academic Department
                  </label>
                  <input
                    type="text"
                    required
                    value={department}
                    onChange={(e) => setDepartment(e.target.value)}
                    onInvalid={(e) => (e.target as HTMLInputElement).setCustomValidity('Enter Department')}
                    onInput={(e) => (e.target as HTMLInputElement).setCustomValidity('')}
                    placeholder="Enter Department"
                    className="w-full bg-slate-50 border border-slate-300 rounded-xl px-4 py-2.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white"
                  />
                </div>
              </>
            )}

            <button
              type="submit"
              className={`w-full py-3 px-4 rounded-xl font-bold text-xs flex items-center justify-center gap-2 text-white shadow-md transition-all ${
                role === 'student'
                  ? 'bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 shadow-sky-500/20'
                  : 'bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-700 hover:to-cyan-700 shadow-indigo-500/20'
              }`}
            >
              <span>{role === 'student' ? 'Enter Guided Workspace' : 'Open Instructor Studio'}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>
        </div>

        {/* Footer info */}
        <p className="text-[11px] text-slate-400 text-center">
          Empowering computer science education with transparent, code-grounded AI assessment.
        </p>
      </div>
    </div>
  );
};
