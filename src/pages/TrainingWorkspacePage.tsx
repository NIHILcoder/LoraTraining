import React, { useState, useCallback, useRef } from 'react';
import { DatasetSection } from '../components/workspace/DatasetSection';
import { ConfigSection } from '../components/workspace/ConfigSection';
import { TrainingMonitor } from '../components/workspace/TrainingMonitor';
import './TrainingWorkspacePage.css';
// Reuse existing page styles for sub-components
import './DatasetPage.css';
import './ConfigPage.css';
import './TrainingPage.css';

export function TrainingWorkspacePage() {
  const [isTraining, setIsTraining] = useState(false);
  const [leftWidth, setLeftWidth] = useState(420);
  const isDragging = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(420);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    startX.current = e.clientX;
    startWidth.current = leftWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const handleMouseMove = (ev: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = ev.clientX - startX.current;
      const newWidth = Math.min(560, Math.max(320, startWidth.current + delta));
      setLeftWidth(newWidth);
    };

    const handleMouseUp = () => {
      isDragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [leftWidth]);

  return (
    <div className="workspace animate-fade-in-up">
      <div className="workspace__panels">
        {/* Left Panel: Dataset + Config */}
        <div className="workspace__left" style={{ width: `${leftWidth}px` }}>
          <div className="workspace__left-inner">
            <DatasetSection />
            <ConfigSection disabled={isTraining} />
          </div>
        </div>

        {/* Resizable Divider */}
        <div
          className={`workspace__divider ${isDragging.current ? 'workspace__divider--dragging' : ''}`}
          onMouseDown={handleMouseDown}
        />

        {/* Right Panel: Training Monitor */}
        <div className="workspace__right">
          <TrainingMonitor onTrainingStateChange={setIsTraining} />
        </div>
      </div>
    </div>
  );
}
