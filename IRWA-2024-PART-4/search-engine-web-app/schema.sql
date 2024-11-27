CREATE TABLE IF NOT EXISTS session (
    session_id INTEGER PRIMARY KEY,
    ip TEXT,
    user_agent TEXT,
    start_time DATETIME,
    end_time DATETIME
);

CREATE TABLE IF NOT EXISTS query (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    query TEXT,
    timestamp DATETIME,
    FOREIGN KEY (session_id) REFERENCES session(session_id)
);

CREATE TABLE IF NOT EXISTS click (
    click_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    doc_id INTEGER,
    timestamp DATETIME,
    FOREIGN KEY (session_id) REFERENCES session(session_id)
);

CREATE TABLE IF NOT EXISTS analytics (
    session_id INTEGER,                     -- ID de la sesión
    query TEXT,                             -- Consulta realizada
    doc_id INTEGER,                         -- ID del documento clicado
    title TEXT,                             -- Título del documento clicado
    description TEXT,                       -- Descripción del documento clicado
    timestamp DATETIME,                     -- Hora del clic o final de la sesión
    browser TEXT,                           -- Navegador utilizado
    operating_system TEXT,                  -- Sistema operativo
    ip_address TEXT                         -- Dirección IP del usuario
);
