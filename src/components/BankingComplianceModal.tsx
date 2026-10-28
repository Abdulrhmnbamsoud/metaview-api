// src/components/BankingComplianceModal.tsx
import { X, ShieldCheck, Lock, FileText, CheckCircle2 } from 'lucide-react';

interface BankingComplianceModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const BankingComplianceModal: React.FC<BankingComplianceModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="bg-white dark:bg-[#0f172a] rounded-2xl border border-stone-200/90 dark:border-slate-800 shadow-2xl max-w-2xl w-full max-h-[90vh] flex flex-col overflow-hidden text-right">
        {/* Modal Header */}
        <div className="p-4 sm:p-5 border-b border-stone-200/80 dark:border-slate-800 flex items-center justify-between bg-stone-50/50 dark:bg-slate-900/40">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-blue-600/10 text-blue-600 dark:text-blue-400 flex items-center justify-center">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-stone-900 dark:text-white">
                سياسة الخصوصية وضوابط البيئة المصرفية
              </h2>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-stone-400 hover:text-stone-700 dark:hover:text-stone-200 hover:bg-stone-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 sm:p-6 overflow-y-auto space-y-5 text-xs text-stone-700 dark:text-slate-300 leading-relaxed">
          {/* Section 1: Nature of Information */}
          <div className="p-3.5 rounded-xl bg-blue-50/50 dark:bg-blue-950/20 border border-blue-200/60 dark:border-blue-900/40 space-y-1.5">
            <div className="flex items-center gap-2 font-bold text-blue-900 dark:text-blue-200 text-sm">
              <FileText className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
              <span>1. طبيعة المعلومات العامة المنشورة</span>
            </div>
            <p>
              منصة <strong>MetaView 3</strong> هي منظومة رصد وتحليل استباقي تعتمد في تغذيتها كلياً على مصادر علنية ومفتوحة تشمل:
              البيانات الصحفية الصادرة عن الجهات الأمنية والقضائية، التحذيرات الرسمية المنشورة من البنوك وشركات المدفوعات، وحملات التوعية المعتمدة. لا تتضمن المنصة أي بيانات غير معلنة أو مصادر غير مرخصة.
            </p>
          </div>

          {/* Section 2: Non-Collection of Confidential Banking Data */}
          <div className="p-3.5 rounded-xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200/60 dark:border-emerald-900/40 space-y-1.5">
            <div className="flex items-center gap-2 font-bold text-emerald-900 dark:text-emerald-200 text-sm">
              <Lock className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
              <span>2. سرية الحسابات وعدم معالجة البيانات الشخصية (PII)</span>
            </div>
            <p>
              تلتزم المنصة بضوابط السرية المصرفية التامة:
            </p>
            <ul className="list-disc pr-5 space-y-1 text-emerald-950 dark:text-emerald-200/90">
              <li>لا يتم جمع، تخزين، أو معالجة أي أرقام حسابات بنكية، آيبان (IBAN)، أو بطاقات دفع حقيقية لعملاء البنوك.</li>
              <li>كافة المؤشرات والإحصاءات المعروضة في المنصة هي بيانات تحليلية عامة مبنية على وقائع منشورة علناً.</li>
              <li>المنصة لا ترتبط بقواعد بيانات العملاء الداخلية لأي بنك أو مؤسسة مالية.</li>
            </ul>
          </div>

          {/* Section 3: Public View-Only Mode */}
          <div className="p-3.5 rounded-xl bg-stone-100/70 dark:bg-slate-800/50 border border-stone-200 dark:border-slate-700/60 space-y-1.5">
            <div className="flex items-center gap-2 font-bold text-stone-900 dark:text-slate-200 text-sm">
              <CheckCircle2 className="w-4 h-4 text-sky-600 dark:text-sky-400 shrink-0" />
              <span>3. وضع المستعرض العام (Public Read-Only Mode)</span>
            </div>
            <p>
              تتيح المنصة للمختصين استعراض كافة التحليلات، الإحصاءات العامة، ومؤشرات الأنماط بصيغة القراءة فقط دون الحاجة إلى منحهم صلاحيات إدارية. تظل أدوات الإدارة وتشغيل الجلب محمية ومحصورة في الخوادم الخلفية.
            </p>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-stone-200/80 dark:border-slate-800 flex items-center justify-between bg-stone-50/50 dark:bg-slate-900/40">
          <span className="text-[11px] text-stone-400 font-mono">
            MetaView 3 Compliance Policy • Ver 3.0
          </span>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-xs transition-colors"
          >
            إغلاق ومتابعة الاستعراض
          </button>
        </div>
      </div>
    </div>
  );
};
