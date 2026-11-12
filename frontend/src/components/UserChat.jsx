import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, Loader2 } from 'lucide-react';
import { sendChatMessage } from '../api';

export default function UserChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! Ask me any question based on the uploaded documents.',
      sources: []
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSubmit = async (e) => {
    e?.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    // Append user message
    setMessages(prev => [...prev, { role: 'user', content: query }]);
    setInput('');
    setLoading(true);

    // Set up AbortController with 60.5s client-side safety timeout
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, 60500);

    try {
      const response = await sendChatMessage(query, controller.signal);
      clearTimeout(timeoutId);

      if (response.status === 'timeout' || response.answer.includes('Taking too long')) {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: 'Taking too long. Please try again.',
            sources: []
          }
        ]);
      } else {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: response.answer,
            sources: response.sources || []
          }
        ]);
      }
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: 'Taking too long. Please try again.',
            sources: []
          }
        ]);
      } else {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: 'An error occurred while fetching the answer. Please try again.',
            sources: []
          }
        ]);
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full px-4 sm:px-6">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto py-6 space-y-6">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex flex-col ${
              msg.role === 'user' ? 'items-end' : 'items-start'
            }`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 leading-relaxed text-[15px] whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-[#2f2f2f] text-white rounded-br-none shadow-sm'
                  : 'bg-transparent text-gray-200'
              }`}
            >
              {msg.content}
            </div>

            {/* Sources under answer */}
            {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
              <div className="mt-1.5 text-xs text-gray-400 pl-1 flex flex-wrap gap-2">
                <span>Sources:</span>
                {msg.sources.map((src, sIdx) => (
                  <span
                    key={sIdx}
                    className="inline-block bg-[#2f2f2f] px-2 py-0.5 rounded text-gray-400 text-[11px]"
                  >
                    {src.file_name} (p. {src.page})
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-center space-x-2 text-gray-400 text-sm py-2">
            <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
            <span>Thinking...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="pb-6 pt-2">
        <form
          onSubmit={handleSubmit}
          className="relative flex items-center bg-[#2f2f2f] rounded-2xl border border-gray-700/60 shadow-lg focus-within:border-gray-500 transition-colors"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your uploaded PDFs..."
            rows={1}
            disabled={loading}
            className="w-full bg-transparent text-white placeholder-gray-400 py-3.5 pl-4 pr-12 focus:outline-none resize-none text-[15px]"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className={`absolute right-2.5 p-2 rounded-xl transition-all ${
              input.trim() && !loading
                ? 'bg-white text-black hover:bg-gray-200'
                : 'bg-gray-600/40 text-gray-500 cursor-not-allowed'
            }`}
          >
            <ArrowUp className="w-4 h-4 stroke-[2.5]" />
          </button>
        </form>
      </div>
    </div>
  );
}
