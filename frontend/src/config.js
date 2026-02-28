const getApiBaseUrl = () => {
    if (process.env.NODE_ENV === 'production') {
        return '';
    }
    return 'http://localhost:8000';
};

const getWsBaseUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    if (process.env.NODE_ENV === 'production') {
        return `${protocol}//${window.location.host}`;
    }
    return `ws://localhost:8000`;
};

export const API_BASE_URL = getApiBaseUrl();
export const WS_BASE_URL = getWsBaseUrl();