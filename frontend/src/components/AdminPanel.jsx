import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, Trash2, LogOut, CheckCircle, Clock, AlertCircle, FileText, Loader2 } from 'lucide-react';
import { loginAdmin, getPdfs, uploadPdf, deletePdf } from '../api';

export default function AdminPanel() {
  const [token, setToken] = useState(localStorage.getItem('admin_token') || '');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);

  const [pdfs, setPdfs] = useState([]);
  const [loadingPdfs, setLoadingPdfs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [dragOver, setDragOver] = useState(false);

  const fileInputRef = useRef(null);

  // Check and fetch PDFs if token exists
  useEffect(() => {
    if (token) {
      loadPdfs();
    }
  }, [token]);

  // Polling when any document is still "Processing"
  useEffect(() => {
    if (!token) return;
    const hasProcessing = pdfs.some(p => p.status === 'Processing');
    if (!hasProcessing) return;

    const interval = setInterval(() => {
      loadPdfs(true);
    }, 3000);

    return () => clearInterval(interval);
  }, [pdfs, token]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    setLoginLoading(true);
    try {
      const data = await loginAdmin(username, password);
      localStorage.setItem('admin_token', data.access_token);
      setToken(data.access_token);
      setUsername('');
      setPassword('');
    } catch (err) {
      setLoginError(err.message || 'Login failed');
    } finally {
      setLoginLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    setToken('');
    setPdfs([]);
  };

  const loadPdfs = async (isBackground = false) => {
    if (!isBackground) setLoadingPdfs(true);
    try {
      const res = await getPdfs(token);
      setPdfs(res.documents || []);
    } catch (err) {
      if (err.message.includes('expired') || err.message.includes('401')) {
        handleLogout();
      }
    } finally {
      if (!isBackground) setLoadingPdfs(false);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Only PDF files are supported.');
      return;
    }
    setUploadError('');
    setUploading(true);
    try {
      await uploadPdf(file, token);
      await loadPdfs();
    } catch (err) {
      setUploadError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleDelete = async (docId) => {
    if (!window.confirm('Are you sure you want to delete this PDF?')) return;
    try {
      await deletePdf(docId, token);
      setPdfs(prev => prev.filter(p => p.id !== docId));
    } catch (err) {
      alert(err.message || 'Failed to delete PDF');
    }
  };

  // Login View
  if (!token) {
    return (
      <div className="flex items-center justify-center h-full px-4">
        <div className="w-full max-w-sm bg-[#2f2f2f] border border-gray-700/60 rounded-2xl p-6 shadow-xl">
          <h2 className="text-xl font-semibold mb-6 text-white text-center">Admin Login</h2>
          {loginError && (
            <div className="mb-4 text-xs bg-red-900/40 text-red-300 border border-red-800 rounded-lg p-2.5">
              {loginError}
            </div>
          )}
          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="w-full bg-[#212121] border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-gray-500"
              />
            </div>
            <div>
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full bg-[#212121] border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-gray-500"
              />
            </div>
            <button
              type="submit"
              disabled={loginLoading}
              className="w-full bg-white text-black font-medium py-2.5 rounded-xl hover:bg-gray-200 transition-colors text-sm flex items-center justify-center"
            >
              {loginLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Log In'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // Admin Dashboard View
  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full px-4 sm:px-6 py-6 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between pb-6 border-b border-gray-800">
        <h2 className="text-xl font-semibold text-white">Document Management</h2>
        <button
          onClick={handleLogout}
          className="flex items-center space-x-1.5 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <LogOut className="w-4 h-4" />
          <span>Logout</span>
        </button>
      </div>

      {/* Drag & Drop Upload */}
      <div className="my-6">
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-colors ${
            dragOver
              ? 'border-white bg-[#2f2f2f]'
              : 'border-gray-700 hover:border-gray-500 bg-[#262626]'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => handleFileUpload(e.target.files?.[0])}
            accept=".pdf"
            className="hidden"
          />
          <div className="flex flex-col items-center justify-center space-y-2">
            {uploading ? (
              <>
                <Loader2 className="w-8 h-8 animate-spin text-gray-300" />
                <p className="text-sm text-gray-300">Uploading PDF & scheduling ingestion...</p>
              </>
            ) : (
              <>
                <UploadCloud className="w-9 h-9 text-gray-400" />
                <p className="text-sm font-medium text-gray-200">
                  Drag and drop a PDF here, or <span className="text-blue-400 underline">browse</span>
                </p>
                <p className="text-xs text-gray-500">PDFs are processed in the background (text, tables & OCR)</p>
              </>
            )}
          </div>
        </div>

        {uploadError && (
          <p className="mt-2 text-xs text-red-400">{uploadError}</p>
        )}
      </div>

      {/* Documents Table */}
      <div className="flex-1">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-300">Uploaded PDFs ({pdfs.length})</h3>
          {loadingPdfs && <Loader2 className="w-3.5 h-3.5 animate-spin text-gray-400" />}
        </div>

        <div className="bg-[#262626] border border-gray-800 rounded-xl overflow-hidden shadow-sm">
          {pdfs.length === 0 ? (
            <div className="p-8 text-center text-sm text-gray-500">
              No PDFs uploaded yet. Upload your first document above.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-[#1f1f1f] text-gray-400 border-b border-gray-800 text-xs uppercase font-medium">
                  <tr>
                    <th className="py-3 px-4">Name</th>
                    <th className="py-3 px-4">Upload Date</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/60">
                  {pdfs.map((doc) => (
                    <tr key={doc.id} className="hover:bg-[#2c2c2c] transition-colors">
                      <td className="py-3 px-4 text-gray-200 font-medium flex items-center space-x-2">
                        <FileText className="w-4 h-4 text-gray-400 flex-shrink-0" />
                        <span className="truncate max-w-xs">{doc.name}</span>
                      </td>
                      <td className="py-3 px-4 text-gray-400 text-xs">
                        {new Date(doc.upload_date).toLocaleString()}
                      </td>
                      <td className="py-3 px-4">
                        {doc.status === 'Processing' && (
                          <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-900/40 text-yellow-300 border border-yellow-700/50">
                            <Clock className="w-3 h-3 animate-spin" />
                            <span>Processing</span>
                          </span>
                        )}
                        {doc.status === 'Ready' && (
                          <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-900/40 text-emerald-300 border border-emerald-700/50">
                            <CheckCircle className="w-3 h-3" />
                            <span>Ready</span>
                          </span>
                        )}
                        {doc.status === 'Failed' && (
                          <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-900/40 text-red-300 border border-red-700/50" title={doc.error_message || ''}>
                            <AlertCircle className="w-3 h-3" />
                            <span>Failed</span>
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={() => handleDelete(doc.id)}
                          className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-red-900/20 rounded-lg transition-colors"
                          title="Delete PDF"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
