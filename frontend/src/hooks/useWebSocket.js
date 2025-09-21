import { useEffect, useRef, useCallback, useState } from 'react';
import { wsManager } from '../services/api';

/**
 * Custom hook for managing WebSocket connections and event listeners
 * Prevents duplicate listeners and handles cleanup properly
 */
export const useWebSocket = (eventHandlers = {}) => {
  const handlersRef = useRef({});
  const isInitialized = useRef(false);

  // Update handlers ref when handlers change
  useEffect(() => {
    handlersRef.current = eventHandlers;
  }, [eventHandlers]);

  // Initialize WebSocket connection only once
  useEffect(() => {
    if (!isInitialized.current) {
      console.log('🔗 Initializing WebSocket connection');
      wsManager.connect();
      isInitialized.current = true;
    }

    // Set up event listeners
    const wrappedHandlers = {};
    
    Object.entries(eventHandlers).forEach(([event, handler]) => {
      const wrappedHandler = (...args) => {
        if (handlersRef.current[event]) {
          handlersRef.current[event](...args);
        }
      };
      
      wrappedHandlers[event] = wrappedHandler;
      wsManager.on(event, wrappedHandler);
    });

    // Cleanup function
    return () => {
      Object.entries(wrappedHandlers).forEach(([event, handler]) => {
        wsManager.off(event, handler);
      });
    };
  }, []); // Empty dependency array - only run once

  // Connection management functions
  const connect = useCallback(() => {
    wsManager.connect();
  }, []);

  const disconnect = useCallback(() => {
    wsManager.disconnect();
  }, []);

  const send = useCallback((type, payload) => {
    wsManager.send(type, payload);
  }, []);

  return {
    connect,
    disconnect,
    send,
    isConnected: wsManager.isConnected
  };
};

/**
 * Hook for getting WebSocket connection status
 */
export const useWebSocketStatus = () => {
  const [status, setStatus] = useState('disconnected');

  useEffect(() => {
    const updateStatus = (newStatus) => setStatus(newStatus);

    wsManager.on('connected', () => updateStatus('connected'));
    wsManager.on('disconnected', () => updateStatus('disconnected'));
    wsManager.on('connecting', () => updateStatus('connecting'));
    wsManager.on('reconnecting', () => updateStatus('reconnecting'));
    wsManager.on('error', () => updateStatus('error'));

    // Initial status
    setStatus(wsManager.isConnected ? 'connected' : 'disconnected');

    return () => {
      wsManager.off('connected', updateStatus);
      wsManager.off('disconnected', updateStatus);
      wsManager.off('connecting', updateStatus);
      wsManager.off('reconnecting', updateStatus);
      wsManager.off('error', updateStatus);
    };
  }, []);

  return status;
};