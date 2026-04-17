import React, { useState } from 'react';
import { Minus, Square, X, Sparkles } from 'lucide-react';
import './TitleBar.css';

export function TitleBar() {
  const [isMaximized, setIsMaximized] = useState(false);

  const handleMinimize = () => {
    if (window.require) {
      const { ipcRenderer } = window.require('electron');
      ipcRenderer.send('window-min');
    }
  };

  const handleMaximize = () => {
    if (window.require) {
      const { ipcRenderer } = window.require('electron');
      ipcRenderer.send('window-max');
      setIsMaximized(!isMaximized);
    }
  };

  const handleClose = () => {
    if (window.require) {
      const { ipcRenderer } = window.require('electron');
      ipcRenderer.send('window-close');
    }
  };

  return (
    <div className="titlebar">
      <div className="titlebar__drag-region">
        <Sparkles size={14} className="titlebar__icon" />
        <span className="titlebar__title">LoRA Training Dashboard</span>
      </div>
      <div className="titlebar__controls">
        <button className="titlebar__btn" onClick={handleMinimize} title="Minimize">
          <Minus size={16} />
        </button>
        <button className="titlebar__btn" onClick={handleMaximize} title="Maximize">
          <Square size={14} />
        </button>
        <button className="titlebar__btn titlebar__btn--close" onClick={handleClose} title="Close">
          <X size={16} />
        </button>
      </div>
    </div>
  );
}
