import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  FlaskConical,
  HardDrive,
  LayoutGrid,
  Sparkles,
  Wifi,
  WifiOff,
  Zap,
} from 'lucide-react';
import { useApp } from '../../context/AppContext';
import './Sidebar.css';

const navItems = [
  { path: '/', label: 'Train', icon: FlaskConical, description: 'Dataset · Config · Run' },
  { path: '/models', label: 'Models Hub', icon: HardDrive, description: 'Download models' },
  { path: '/playground', label: 'Playground', icon: Zap, description: 'Test models' },
  { path: '/gallery', label: 'Gallery', icon: LayoutGrid, description: 'Saved models' },
];

export function Sidebar() {
  const { state } = useApp();
  const location = useLocation();

  return (
    <aside className="sidebar" id="main-sidebar">
      {/* Logo */}
      <div className="sidebar__brand">
        <div className="sidebar__logo">
          <Sparkles size={22} />
        </div>
        <div className="sidebar__brand-text">
          <span className="sidebar__brand-name">LoRA Studio</span>
          <span className="sidebar__brand-version">v1.0.0</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar__nav">
        <span className="sidebar__nav-label">Navigation</span>
        <ul className="sidebar__menu">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
                }
                end={item.path === '/'}
              >
                <span className="sidebar__link-icon">
                  <item.icon size={20} />
                </span>
                <div className="sidebar__link-text">
                  <span className="sidebar__link-label">{item.label}</span>
                  <span className="sidebar__link-desc">{item.description}</span>
                </div>
                {item.path === '/' && state.trainingStatus.phase === 'training' && (
                  <span className="sidebar__live-dot" />
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Status Footer */}
      <div className="sidebar__footer">
        <div className="sidebar__status">
          {state.wsConnected ? (
            <>
              <Wifi size={14} className="sidebar__status-icon sidebar__status-icon--connected" />
              <span>Backend Connected</span>
            </>
          ) : (
            <>
              <WifiOff size={14} className="sidebar__status-icon sidebar__status-icon--disconnected" />
              <span>Offline Mode</span>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
