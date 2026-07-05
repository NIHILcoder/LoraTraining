import React, { useEffect, useState } from 'react';
import { Download, RefreshCw } from 'lucide-react';

type UpdateEvent =
  | { type: 'checking' }
  | { type: 'available'; version?: string }
  | { type: 'not-available' }
  | { type: 'progress'; percent: number; bytesPerSecond?: number; transferred?: number; total?: number }
  | { type: 'downloaded'; version?: string }
  | { type: 'error'; message?: string };

/**
 * Listens for auto-update events from the main process (electron-updater + GitHub Releases)
 * and shows an unobtrusive banner. Silent while checking / up-to-date, and on errors
 * (which are expected in dev or before a published release with latest.yml exists).
 */
export function UpdateNotification() {
  const [event, setEvent] = useState<UpdateEvent | null>(null);

  useEffect(() => {
    const api = window.loraStudio;
    if (!api?.on) return;
    const handler = (data: UpdateEvent) => setEvent(data);
    api.on('update-event', handler);
    // Ask the main process to check now (it no-ops in dev / unpackaged builds).
    api.checkForUpdates?.().catch(() => { /* surfaced via the error event, which we ignore in UI */ });
    return () => api.removeAllListeners?.('update-event');
  }, []);

  if (!event) return null;
  if (event.type !== 'available' && event.type !== 'progress' && event.type !== 'downloaded') {
    return null;
  }

  const wrap: React.CSSProperties = {
    position: 'fixed', bottom: 16, right: 16, zIndex: 9999,
    minWidth: 260, maxWidth: 360, padding: '12px 14px',
    background: 'var(--color-bg-elevated, #1a1a2e)',
    border: '1px solid var(--color-surface-border, #2a2a4a)',
    borderRadius: 'var(--radius-md, 8px)',
    boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
    color: 'var(--color-text-primary, #fff)', fontSize: 13,
    display: 'flex', flexDirection: 'column', gap: 8,
  };
  const row: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 8 };

  return (
    <div style={wrap}>
      {event.type === 'available' && (
        <div style={row}>
          <Download size={16} color="var(--color-accent, #7c6af5)" />
          <span>Update {event.version ? `v${event.version}` : ''} available — downloading…</span>
        </div>
      )}

      {event.type === 'progress' && (
        <>
          <div style={row}>
            <RefreshCw size={16} className="animate-spin" color="var(--color-accent, #7c6af5)" />
            <span>Downloading update… {event.percent}%</span>
          </div>
          <div style={{ height: 4, background: 'var(--color-bg-primary, #0f0f1a)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ width: `${event.percent}%`, height: '100%', background: 'var(--color-accent, #7c6af5)' }} />
          </div>
        </>
      )}

      {event.type === 'downloaded' && (
        <>
          <div style={row}>
            <Download size={16} color="var(--color-success, #22c55e)" />
            <span>Update {event.version ? `v${event.version}` : ''} ready to install.</span>
          </div>
          <button
            onClick={() => window.loraStudio?.installUpdate?.()}
            style={{
              alignSelf: 'flex-start', marginTop: 2, padding: '6px 12px',
              background: 'var(--color-accent, #7c6af5)', color: '#fff',
              border: 'none', borderRadius: 'var(--radius-sm, 6px)', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            }}
          >
            Restart &amp; Install
          </button>
        </>
      )}
    </div>
  );
}
