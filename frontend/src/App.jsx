import React, { useState } from 'react';
import { MessageSquare, ShieldCheck } from 'lucide-react';
import UserChat from './components/UserChat';
import AdminPanel from './components/AdminPanel';

export default function App() {
  const [activeTab, setActiveTab] = useState('user'); // 'user' or 'admin'

  return (
    <div className="flex flex-col h-screen w-screen bg-[#212121] text-gray-100 select-none">
      {/* Minimal Header */}
      <header className="h-14 border-b border-gray-800/80 px-4 sm:px-6 flex items-center justify-between flex-shrink-0 bg-[#171717]/60 backdrop-blur">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-white tracking-tight text-[15px]">
            PDF Q&A Assistant
          </span>
        </div>

        {/* Minimal Role Switcher */}
        <div className="flex items-center bg-[#282828] p-1 rounded-xl border border-gray-700/50">
          <button
            onClick={() => setActiveTab('user')}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'user'
                ? 'bg-[#383838] text-white shadow-sm'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Chat</span>
          </button>
          <button
            onClick={() => setActiveTab('admin')}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'admin'
                ? 'bg-[#383838] text-white shadow-sm'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Admin</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 overflow-hidden">
        {activeTab === 'user' ? <UserChat /> : <AdminPanel />}
      </main>
    </div>
  );
}
