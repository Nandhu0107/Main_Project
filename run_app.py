import socket
import uvicorn


HOST = "127.0.0.1"
PORT_CANDIDATES = [8000, 8001, 8002, 8003, 8004]


def is_port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def choose_port():
    for port in PORT_CANDIDATES:
        if is_port_free(HOST, port):
            return port
    raise RuntimeError(f"No free port found in {PORT_CANDIDATES}")


def main():
    port = choose_port()
    print(f"Starting app at http://{HOST}:{port}")
    uvicorn.run("app:asgi_app", host=HOST, port=port, lifespan="off")


if __name__ == "__main__":
    main()
