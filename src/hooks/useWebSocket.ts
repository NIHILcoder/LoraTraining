import { useEffect, useRef, useState, useCallback } from 'react';
import type { WSMessage } from '../types';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (message: WSMessage) => void;
  reconnectInterval?: number;
  /** Default: reconnect forever. Pass a finite number to cap attempts. */
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
  maxRetries = Number.POSITIVE_INFINITY,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const isMountedRef = useRef(false);
  const urlRef = useRef(url);
  urlRef.current = url;

  const onMessageRef = useRef(onMessage);
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  const connect = useCallback(() => {
    const state = wsRef.current?.readyState;
    if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return;
    if (!isMountedRef.current) return;
    const target = urlRef.current;
    if (!target) return;

    try {
      const ws = new WebSocket(target);

      ws.onopen = () => {
        if (!isMountedRef.current) { ws.close(); return; }
        setIsConnected(true);
        retriesRef.current = 0;
        console.log('[WS] Connected to', target);
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
          retriesRef.current = Math.min(retriesRef.current + 1, 20);
          const delay = Math.min(
            reconnectInterval * Math.pow(1.5, retriesRef.current - 1),
            30000
          );
          reconnectTimerRef.current = setTimeout(() => {
            if (isMountedRef.current) connect();
          }, delay);
        }
      };

      ws.onerror = (err) => {
        console.error('[WS] Error:', err);
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('[WS] Connection failed:', err);
    }
  }, [reconnectInterval, maxRetries]);

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
    wsRef.current = null;
    retriesRef.current = 0;
    connect();
  }, [connect]);

  useEffect(() => {
    isMountedRef.current = true;
    retriesRef.current = 0;
    connect();
    return () => {
      isMountedRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [url, connect]);

  return { isConnected, lastMessage, send, reconnect };
}
