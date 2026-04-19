import { useEffect, useRef, useState, useCallback } from 'react';
import type { WSMessage } from '../types';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WSMessage) => void;
  reconnectInterval?: number;
  maxRetries?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: WSMessage | null;
  send: (data: unknown) => void;
  reconnect: () => void;
}

export function useWebSocket({
  url,
  onMessage,
  reconnectInterval = 3000,
  maxRetries = 10,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const isMountedRef = useRef(false);

  // Keep onMessage stable via ref so it doesn't re-trigger effects
  const onMessageRef = useRef(onMessage);
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  const connect = useCallback(() => {
    // Don't open a new connection if one is already open or connecting
    const state = wsRef.current?.readyState;
    if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return;
    if (!isMountedRef.current) return;

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        if (!isMountedRef.current) { ws.close(); return; }
        setIsConnected(true);
        retriesRef.current = 0;
        console.log('[WS] Connected to', url);
      };

      ws.onmessage = (event) => {
        if (!isMountedRef.current) return;
        try {
          const message: WSMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessageRef.current?.(message);
        } catch (err) {
          console.error('[WS] Failed to parse message:', err);
        }
      };

      ws.onclose = () => {
        if (!isMountedRef.current) return;
        setIsConnected(false);
        console.log('[WS] Disconnected');

        if (retriesRef.current < maxRetries && isMountedRef.current) {
          retriesRef.current += 1;
          reconnectTimerRef.current = setTimeout(() => {
            if (isMountedRef.current) connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (err) => {
        console.error('[WS] Error:', err);
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('[WS] Connection failed:', err);
    }
  }, [url, reconnectInterval, maxRetries]); // stable deps only

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('[WS] Cannot send — not connected');
    }
  }, []);

  const reconnect = useCallback(() => {
    clearTimeout(reconnectTimerRef.current);
    wsRef.current?.close();
    retriesRef.current = 0;
    connect();
  }, [connect]);

  useEffect(() => {
    isMountedRef.current = true;
    connect();
    return () => {
      isMountedRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []); // only run on mount/unmount — connect is stable via useCallback

  return { isConnected, lastMessage, send, reconnect };
}
