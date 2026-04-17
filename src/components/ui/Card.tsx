import React from 'react';
import './Card.css';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hover?: boolean;
  glow?: boolean;
  onClick?: () => void;
  style?: React.CSSProperties;
}

export function Card({
  children,
  className = '',
  padding = 'md',
  hover = false,
  glow = false,
  onClick,
  style,
}: CardProps) {
  return (
    <div
      className={`card card--pad-${padding} ${hover ? 'card--hover' : ''} ${glow ? 'card--glow' : ''} ${onClick ? 'card--clickable' : ''} ${className}`}
      style={style}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {children}
    </div>
  );
}
