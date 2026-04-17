import React from 'react';
import './Badge.css';

interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info' | 'accent';
  size?: 'sm' | 'md';
  dot?: boolean;
  children: React.ReactNode;
}

export function Badge({
  variant = 'default',
  size = 'sm',
  dot = false,
  children,
}: BadgeProps) {
  return (
    <span className={`badge badge--${variant} badge--${size}`}>
      {dot && <span className="badge__dot" />}
      {children}
    </span>
  );
}
