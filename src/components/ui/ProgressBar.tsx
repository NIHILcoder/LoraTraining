import React from 'react';
import './ProgressBar.css';

interface ProgressBarProps {
  value: number;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'accent' | 'success' | 'warning' | 'error';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
}

export function ProgressBar({
  value,
  max = 100,
  size = 'md',
  variant = 'accent',
  showLabel = false,
  label,
  animated = true,
}: ProgressBarProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  return (
    <div className={`progress progress--${size}`}>
      {(showLabel || label) && (
        <div className="progress__header">
          {label && <span className="progress__label">{label}</span>}
          {showLabel && (
            <span className="progress__value">{percentage.toFixed(1)}%</span>
          )}
        </div>
      )}
      <div className="progress__track">
        <div
          className={`progress__fill progress__fill--${variant} ${animated ? 'progress__fill--animated' : ''}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
