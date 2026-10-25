// src/pages/TeamManagement.tsx
import React, { useState, useEffect } from 'react';
import {
  Users,
  UserPlus,
  Shield,
  ShieldAlert,
  Clock,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  UserX,
  RefreshCw,
  Search,
  Filter,
  Eye,
  FileText,
  Key,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { User, UserRole, UserStatus, Invitation, AuditLog } from '../types/auth';
import { fmtDateTime } from '../components/ui';

export const TeamManagement: React.FC = () => {
  const { user: currentUser, can } = useAuth();

  const [activeTab, setActiveTab] = useState<'members' | 'invitations' | 'audit'>('members');
  const [members, setMembers] = useState<User[]>([]);
  const [invitations, setInvitations] = useState<Invitation[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  // Modals & Action States
  const [inviteModalOpen, setInviteModalOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<UserRole>('analyst');
  const [inviteResult, setInviteResult] = useState<{ url: string; token: string } | null>(null);
  const [copied, setCopied] = useState(false);

  const [roleModalUser, setRoleModalUser] = useState<User | null>(null);
  const [selectedNewRole, setSelectedNewRole] = useState<UserRole>('analyst');

  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    setActionError(null);
    try {
      const [membersRes, invitesRes, logsRes] = await Promise.all([
        apiService.getTeamMembers(),
        apiService.getTeamInvitations(),
        apiService.getTeamAuditLogs(100),
      ]);
      setMembers(membersRes.members || []);
      setInvitations(invitesRes.invitations || []);
      setAuditLogs(logsRes.logs || []);
    } catch (err: any) {
      setActionError(err.message || 'تعذر تحميل بيانات الفريق.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (can('team.manage')) {
      fetchData();
    }
  }, [currentUser]);

  if (!can('team.manage')) {
    return (
      <div className="max-w-2xl mx-auto my-12 p-8 bg-white dark:bg-[#0f172a] rounded-2xl border border-red-200 dark:border-red-900/60 text-center space-y-4 shadow-sm">
        <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-950/60 text-red-600 dark:text-red-400 mx-auto flex items-center justify-center">
          <ShieldAlert className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-bold text-stone-900 dark:text-white">
          غير مصرح بالدخول — إدارة الفريق
        </h2>
        <p className="text-xs text-stone-500 dark:text-slate-400 leading-relaxed">
          صفحة إدارة الفريق وصلاحيات الأعضاء مخصصة لمدير المنصة (ADMIN) فقط وفق مبدأ الحد الأدنى من الصلاحيات (Least Privilege).
        </p>
      </div>
    );
  }

  const handleSendInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    setActionError(null);
    setSubmitting(true);
    try {
      const res = await apiService.inviteMember({
        email: inviteEmail.trim().toLowerCase(),
        role: inviteRole,
      });
      setInviteResult({ url: res.invite_url, token: res.invite_token });
      setActionSuccess(`تم إنشاء رابط الدعوة بنجاح لـ ${inviteEmail}.`);
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر إرسال الدعوة.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleRevokeInvite = async (invitationId: number) => {
    setActionError(null);
    try {
      await apiService.revokeInvitation(invitationId);
      setActionSuccess('تم إلغاء الدعوة بنجاح.');
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر إلغاء الدعوة.');
    }
  };

  const handleUpdateRole = async () => {
    if (!roleModalUser) return;
    setActionError(null);
    setSubmitting(true);
    try {
      await apiService.updateMemberRole(roleModalUser.id, selectedNewRole);
      setActionSuccess(`تم تحديث دور ${roleModalUser.full_name} إلى ${getRoleBadge(selectedNewRole).label}.`);
      setRoleModalUser(null);
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر تحديث الدور.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleUpdateStatus = async (user: User, newStatus: UserStatus) => {
    const actionName = newStatus === 'active' ? 'تنشيط' : newStatus === 'suspended' ? 'تعليق' : 'إلغاء وصول';
    if (!confirm(`هل أنت متأكد من ${actionName} حساب ${user.full_name}؟`)) return;

    setActionError(null);
    try {
      await apiService.updateMemberStatus(user.id, newStatus);
      setActionSuccess(`تم تغيير حالة ${user.full_name} بنجاح.`);
      fetchData();
    } catch (err: any) {
      setActionError(err.message || 'تعذر تغيير حالة المستخدم.');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getRoleBadge = (role: UserRole) => {
    switch (role) {
      case 'admin':
        return { label: 'مدير المنصة', color: 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950/40 dark:text-purple-300 dark:border-purple-800' };
      case 'analyst':
        return { label: 'محلل ميتا فيو', color: 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 dark:border-blue-800' };
      case 'viewer':
        return { label: 'مستعرض مصرح', color: 'bg-stone-50 text-stone-700 border-stone-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700' };
    }
  };

  const getStatusBadge = (status: UserStatus) => {
    switch (status) {
      case 'active':
        return { label: 'نشط', color: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800' };
      case 'suspended':
        return { label: 'معلق مؤقتاً', color: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-800' };
      case 'revoked':
        return { label: 'ملغى الوصول', color: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-800' };
      default:
        return { label: status, color: 'bg-stone-100 text-stone-600 dark:bg-slate-800 dark:text-slate-400' };
    }
  };

  const filteredMembers = members.filter(
    (m) =>
      m.full_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white dark:bg-[#0f172a] p-5 sm:p-6 rounded-2xl border border-stone-200/80 dark:border-slate-800 shadow-sm">
        <div className="space-y-1">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400">
              <Users className="w-5 h-5" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-stone-900 dark:text-white">
              إدارة الفريق والصلاحيات
            </h1>
          </div>
          <p className="text-xs text-stone-500 dark:text-slate-400">
            حوكمة وصول المحللين والمدراء، إدارة الدعوات، ومراقبة سجل العمليات المعتمدة
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            onClick={() => {
              setInviteModalOpen(true);
              setInviteResult(null);
            }}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white font-semibold text-xs shadow-sm transition-all"
          >
            <UserPlus className="w-4 h-4" />
            <span>دعوة عضو جديد</span>
          </button>
          <button
            onClick={fetchData}
            className="p-2 rounded-xl border border-stone-200 dark:border-slate-700 text-stone-600 dark:text-slate-400 hover:bg-stone-50 dark:hover:bg-slate-800 transition-colors"
            title="تحديث البيانات"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Notifications */}
      {actionSuccess && (
        <div className="p-3.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800/60 text-emerald-800 dark:text-emerald-200 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
            <span>{actionSuccess}</span>
          </div>
          <button onClick={() => setActionSuccess(null)} className="text-emerald-600 hover:text-emerald-800 text-xs">
            ✕
          </button>
        </div>
      )}

      {actionError && (
        <div className="p-3.5 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/60 text-red-800 dark:text-red-200 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />
            <span>{actionError}</span>
          </div>
          <button onClick={() => setActionError(null)} className="text-red-600 hover:text-red-800 text-xs">
            ✕
          </button>
        </div>
      )}

      {/* Stats Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">إجمالي أعضاء الفريق</div>
          <div className="text-2xl font-bold text-stone-900 dark:text-white font-mono">{members.length}</div>
        </div>
        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">الحسابات النشطة</div>
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 font-mono">
            {members.filter((m) => m.status === 'active').length}
          </div>
        </div>
        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">الدعوات المعلقة</div>
          <div className="text-2xl font-bold text-amber-600 dark:text-amber-400 font-mono">
            {invitations.filter((i) => i.status === 'pending').length}
          </div>
        </div>
        <div className="p-4 rounded-2xl bg-white dark:bg-[#0f172a] border border-stone-200/80 dark:border-slate-800 shadow-sm space-y-1">
          <div className="text-[11px] font-semibold text-stone-400 dark:text-slate-500">حسابات موقوفة / ملغاة</div>
          <div className="text-2xl font-bold text-red-600 dark:text-red-400 font-mono">
            {members.filter((m) => m.status === 'suspended' || m.status === 'revoked').length}
          </div>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex border-b border-stone-200 dark:border-slate-800 gap-4">
        <button
          onClick={() => setActiveTab('members')}
          className={`pb-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
            activeTab === 'members'
              ? 'border-blue-600 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-900'
          }`}
        >
          <Users className="w-4 h-4" />
          <span>أعضاء الفريق ({members.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('invitations')}
          className={`pb-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
            activeTab === 'invitations'
              ? 'border-blue-600 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-900'
          }`}
        >
          <Clock className="w-4 h-4" />
          <span>الدعوات النشطة ({invitations.filter((i) => i.status === 'pending').length})</span>
        </button>

        <button
          onClick={() => setActiveTab('audit')}
          className={`pb-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all ${
            activeTab === 'audit'
              ? 'border-blue-600 text-blue-600 dark:text-blue-400'
              : 'border-transparent text-stone-500 dark:text-slate-400 hover:text-stone-900'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>سجل التدقيق والحوكمة ({auditLogs.length})</span>
        </button>
      </div>

      {/* Tab 1: Team Members Table */}
      {activeTab === 'members' && (
        <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 shadow-sm overflow-hidden space-y-4">
          <div className="p-4 border-b border-stone-100 dark:border-slate-800/80 flex items-center justify-between gap-4">
            <div className="relative flex-1 max-w-sm">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="بحث بالاسم أو البريد الإلكتروني..."
                className="w-full bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 py-1.5 pr-8 pl-3 rounded-xl text-xs text-stone-900 dark:text-white placeholder:text-stone-400 focus:outline-none focus:border-blue-500"
              />
              <Search className="w-3.5 h-3.5 text-stone-400 absolute top-1/2 -translate-y-1/2 right-2.5 pointer-events-none" />
            </div>
            <div className="text-xs text-stone-500 dark:text-slate-400">
              إجمالي المطابق: {filteredMembers.length}
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-right text-xs">
              <thead className="bg-stone-50/70 dark:bg-slate-900/40 text-stone-500 dark:text-slate-400 border-b border-stone-100 dark:border-slate-800">
                <tr>
                  <th className="py-3 px-4 font-semibold">العضو</th>
                  <th className="py-3 px-4 font-semibold">الدور</th>
                  <th className="py-3 px-4 font-semibold">الحالة</th>
                  <th className="py-3 px-4 font-semibold">تاريخ الانضمام</th>
                  <th className="py-3 px-4 font-semibold">آخر تسجيل دخول</th>
                  <th className="py-3 px-4 font-semibold text-center">الإجراءات</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 dark:divide-slate-800/60 text-stone-800 dark:text-stone-200">
                {filteredMembers.map((m) => {
                  const roleBadge = getRoleBadge(m.role);
                  const statusBadge = getStatusBadge(m.status);
                  const isCurrent = currentUser?.id === m.id;

                  return (
                    <tr key={m.id} className="hover:bg-stone-50/60 dark:hover:bg-slate-800/30 transition-colors">
                      <td className="py-3 px-4">
                        <div className="font-semibold text-stone-900 dark:text-white flex items-center gap-1.5">
                          <span>{m.full_name}</span>
                          {isCurrent && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300 font-normal">
                              حسابك
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-stone-400 dark:text-slate-500 font-mono" dir="ltr">
                          {m.email}
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-block px-2 py-0.5 rounded-md border text-[11px] font-semibold ${roleBadge.color}`}>
                          {roleBadge.label}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-block px-2 py-0.5 rounded-md border text-[11px] font-medium ${statusBadge.color}`}>
                          {statusBadge.label}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-stone-500 dark:text-slate-400 font-mono text-[11px]">
                        {fmtDateTime(m.created_at)}
                      </td>
                      <td className="py-3 px-4 text-stone-500 dark:text-slate-400 font-mono text-[11px]">
                        {m.last_login_at ? fmtDateTime(m.last_login_at) : 'لم يسجل دخول بعد'}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center justify-center gap-1.5">
                          {/* Role Change Button */}
                          <button
                            onClick={() => {
                              setRoleModalUser(m);
                              setSelectedNewRole(m.role);
                            }}
                            disabled={isCurrent}
                            className="px-2.5 py-1 rounded-lg border border-stone-200 dark:border-slate-700 text-stone-700 dark:text-slate-300 hover:bg-stone-100 dark:hover:bg-slate-800 disabled:opacity-40 text-[11px] font-medium transition-colors"
                          >
                            تعديل الدور
                          </button>

                          {/* Suspend / Reactivate */}
                          {m.status === 'active' ? (
                            <button
                              onClick={() => handleUpdateStatus(m, 'suspended')}
                              disabled={isCurrent}
                              className="px-2.5 py-1 rounded-lg border border-amber-200 dark:border-amber-800/60 text-amber-700 dark:text-amber-300 hover:bg-amber-50 dark:hover:bg-amber-950/40 disabled:opacity-40 text-[11px] font-medium transition-colors"
                            >
                              تعليق
                            </button>
                          ) : (
                            <button
                              onClick={() => handleUpdateStatus(m, 'active')}
                              className="px-2.5 py-1 rounded-lg border border-emerald-200 dark:border-emerald-800/60 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-50 dark:hover:bg-emerald-950/40 text-[11px] font-medium transition-colors"
                            >
                              تفعيل
                            </button>
                          )}

                          {/* Revoke */}
                          {m.status !== 'revoked' && (
                            <button
                              onClick={() => handleUpdateStatus(m, 'revoked')}
                              disabled={isCurrent}
                              className="p-1 rounded-lg text-red-500 hover:bg-red-50 dark:hover:bg-red-950/40 disabled:opacity-40 transition-colors"
                              title="إلغاء الوصول نهائياً"
                            >
                              <UserX className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Pending Invitations Table */}
      {activeTab === 'invitations' && (
        <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-right text-xs">
              <thead className="bg-stone-50/70 dark:bg-slate-900/40 text-stone-500 dark:text-slate-400 border-b border-stone-100 dark:border-slate-800">
                <tr>
                  <th className="py-3 px-4 font-semibold">البريد الإلكتروني</th>
                  <th className="py-3 px-4 font-semibold">الدور المخصص</th>
                  <th className="py-3 px-4 font-semibold">تاريخ الإرسال</th>
                  <th className="py-3 px-4 font-semibold">تاريخ الانتهاء</th>
                  <th className="py-3 px-4 font-semibold">الحالة</th>
                  <th className="py-3 px-4 font-semibold text-center">الإجراءات</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 dark:divide-slate-800/60 text-stone-800 dark:text-stone-200">
                {invitations.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-stone-400">
                      لا توجد دعوات نشطة حالياً.
                    </td>
                  </tr>
                ) : (
                  invitations.map((inv) => (
                    <tr key={inv.id} className="hover:bg-stone-50/60 dark:hover:bg-slate-800/30">
                      <td className="py-3 px-4 font-mono font-medium" dir="ltr">
                        {inv.email}
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-block px-2 py-0.5 rounded-md border text-[11px] font-semibold ${getRoleBadge(inv.role).color}`}>
                          {getRoleBadge(inv.role).label}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-stone-500 dark:text-slate-400 font-mono text-[11px]">
                        {fmtDateTime(inv.created_at)}
                      </td>
                      <td className="py-3 px-4 text-stone-500 dark:text-slate-400 font-mono text-[11px]">
                        {fmtDateTime(inv.expires_at)}
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-block px-2 py-0.5 rounded-md text-[11px] font-medium ${
                          inv.status === 'pending'
                            ? 'bg-amber-50 text-amber-700 border border-amber-200'
                            : inv.status === 'accepted'
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : 'bg-stone-100 text-stone-600'
                        }`}>
                          {inv.status === 'pending' ? 'قيد الانتظار' : inv.status === 'accepted' ? 'تم القبول' : 'ملغاة'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        {inv.status === 'pending' && (
                          <button
                            onClick={() => handleRevokeInvite(inv.id)}
                            className="px-2 py-1 rounded-lg text-red-600 hover:bg-red-50 dark:hover:bg-red-950/40 text-[11px] font-medium transition-colors"
                          >
                            إلغاء الدعوة
                          </button>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 3: Audit Logs Table */}
      {activeTab === 'audit' && (
        <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/80 dark:border-slate-800 shadow-sm overflow-hidden space-y-4">
          <div className="overflow-x-auto">
            <table className="w-full text-right text-xs">
              <thead className="bg-stone-50/70 dark:bg-slate-900/40 text-stone-500 dark:text-slate-400 border-b border-stone-100 dark:border-slate-800">
                <tr>
                  <th className="py-3 px-4 font-semibold">التاريخ والوقت</th>
                  <th className="py-3 px-4 font-semibold">المستخدم</th>
                  <th className="py-3 px-4 font-semibold">العملية المعتمدة</th>
                  <th className="py-3 px-4 font-semibold">الهدف</th>
                  <th className="py-3 px-4 font-semibold">عنوان IP</th>
                  <th className="py-3 px-4 font-semibold">التفاصيل</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 dark:divide-slate-800/60 text-stone-800 dark:text-stone-200 font-mono text-[11px]">
                {auditLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-stone-50/60 dark:hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 text-stone-500 dark:text-slate-400">{fmtDateTime(log.created_at)}</td>
                    <td className="py-2.5 px-4 font-sans font-medium text-stone-900 dark:text-white">
                      {log.user_name || log.user_email || 'نظام آلي'}
                    </td>
                    <td className="py-2.5 px-4">
                      <span className="px-2 py-0.5 rounded bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-900/60 font-semibold">
                        {log.action}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 text-stone-600 dark:text-slate-300">
                      {log.target_type ? `${log.target_type}:${log.target_id || ''}` : '-'}
                    </td>
                    <td className="py-2.5 px-4 text-stone-400">{log.ip_address || 'unknown'}</td>
                    <td className="py-2.5 px-4 text-[10px] text-stone-500 dark:text-slate-400 max-w-xs truncate" title={JSON.stringify(log.details)}>
                      {log.details ? JSON.stringify(log.details) : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Invite Member Modal */}
      {inviteModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200 dark:border-slate-800 max-w-md w-full p-6 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-stone-100 dark:border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <UserPlus className="w-5 h-5 text-blue-600" />
                <h3 className="font-bold text-sm text-stone-900 dark:text-white">دعوة عضو جديد للمنصة</h3>
              </div>
              <button onClick={() => setInviteModalOpen(false)} className="text-stone-400 hover:text-stone-600">
                ✕
              </button>
            </div>

            {!inviteResult ? (
              <form onSubmit={handleSendInvite} className="space-y-4">
                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                    البريد الإلكتروني للعضو
                  </label>
                  <input
                    type="email"
                    value={inviteEmail}
                    onChange={(e) => setInviteEmail(e.target.value)}
                    placeholder="analyst@bank.sa"
                    dir="ltr"
                    className="w-full px-3.5 py-2.5 rounded-xl bg-stone-50 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 text-xs text-stone-900 dark:text-white placeholder:text-stone-400 focus:outline-none focus:border-blue-500"
                    required
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-stone-700 dark:text-slate-300">
                    الدور المخصص
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {(['analyst', 'viewer', 'admin'] as UserRole[]).map((r) => (
                      <button
                        key={r}
                        type="button"
                        onClick={() => setInviteRole(r)}
                        className={`p-2.5 rounded-xl border text-center transition-all ${
                          inviteRole === r
                            ? 'border-blue-600 bg-blue-50/60 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 font-bold'
                            : 'border-stone-200 dark:border-slate-700 text-stone-600 dark:text-slate-400 hover:bg-stone-50'
                        }`}
                      >
                        <div className="text-xs">{getRoleBadge(r).label}</div>
                        <div className="text-[10px] font-mono opacity-60 uppercase">{r}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-blue-50/60 dark:bg-blue-950/30 text-[11px] text-blue-700 dark:text-blue-300 space-y-1">
                  <div className="font-semibold">ملاحظة أمنية:</div>
                  <div>• يتم إنشاء رمز تشفيري أحادي الاستخدام ينتهي تلقائياً بعد 48 ساعة.</div>
                  <div>• لا يتم حفظ الرمز الخام في قاعدة البيانات؛ يحفظ فقط ناتج الهاش المقاوم للسرقة.</div>
                </div>

                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setInviteModalOpen(false)}
                    className="px-4 py-2 rounded-xl border border-stone-200 dark:border-slate-700 text-xs font-medium text-stone-600 hover:bg-stone-50"
                  >
                    إلغاء
                  </button>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-sm disabled:opacity-50"
                  >
                    {submitting ? 'جارِ الإنشاء...' : 'إنشاء الدعوة'}
                  </button>
                </div>
              </form>
            ) : (
              <div className="space-y-4">
                <div className="p-3 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200 text-xs flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" />
                  <span>تم إنشاء رابط الدعوة بنجاح! شارك الرابط مع العضو لتفعيل حسابه:</span>
                </div>

                <div className="space-y-1">
                  <div className="text-[11px] font-semibold text-stone-500 dark:text-slate-400">رابط الدعوة المباشر:</div>
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      readOnly
                      value={inviteResult.url}
                      className="flex-1 bg-stone-100 dark:bg-slate-900 border border-stone-200 dark:border-slate-700 p-2 rounded-xl text-xs font-mono text-stone-800 dark:text-slate-200"
                    />
                    <button
                      onClick={() => copyToClipboard(inviteResult.url)}
                      className="p-2 rounded-xl bg-blue-600 text-white hover:bg-blue-700 flex items-center gap-1 text-xs shrink-0 font-medium"
                    >
                      {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      <span>{copied ? 'تم النسخ' : 'نسخ'}</span>
                    </button>
                  </div>
                </div>

                <div className="flex justify-end pt-2">
                  <button
                    onClick={() => {
                      setInviteModalOpen(false);
                      setInviteResult(null);
                    }}
                    className="px-4 py-2 rounded-xl bg-stone-900 dark:bg-white text-white dark:text-stone-900 text-xs font-semibold"
                  >
                    إغلاق
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Role Change Modal */}
      {roleModalUser && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200 dark:border-slate-800 max-w-sm w-full p-6 space-y-4 shadow-2xl">
            <h3 className="font-bold text-sm text-stone-900 dark:text-white">
              تعديل دور: {roleModalUser.full_name}
            </h3>

            <div className="space-y-2">
              {(['admin', 'analyst', 'viewer'] as UserRole[]).map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setSelectedNewRole(r)}
                  className={`w-full p-3 rounded-xl border text-right transition-all flex items-center justify-between ${
                    selectedNewRole === r
                      ? 'border-blue-600 bg-blue-50/60 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 font-bold'
                      : 'border-stone-200 dark:border-slate-700 text-stone-600 dark:text-slate-400 hover:bg-stone-50'
                  }`}
                >
                  <div>
                    <div className="text-xs">{getRoleBadge(r).label}</div>
                    <div className="text-[10px] font-mono opacity-60 uppercase">{r}</div>
                  </div>
                  {selectedNewRole === r && <Check className="w-4 h-4 text-blue-600" />}
                </button>
              ))}
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setRoleModalUser(null)}
                className="px-3.5 py-1.5 rounded-xl border border-stone-200 dark:border-slate-700 text-xs font-medium text-stone-600"
              >
                إلغاء
              </button>
              <button
                type="button"
                disabled={submitting || selectedNewRole === roleModalUser.role}
                onClick={handleUpdateRole}
                className="px-4 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold disabled:opacity-50"
              >
                {submitting ? 'جارِ الحفظ...' : 'تأكيد التعديل'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
