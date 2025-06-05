# minimal_server.py
import flwr as fl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MinimalFlowerServer")

SERVER_ADDRESS = "0.0.0.0:8080" # Or try "127.0.0.1:8080" or a different port like 8089

try:
    logger.info(f"Attempting to start MINIMAL Flower server on {SERVER_ADDRESS} for 1 round...")
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=fl.server.strategy.FedAvg() # Simplest possible strategy
    )
    logger.info("Minimal Flower server completed its 1 round.")
except Exception as e:
    logger.error(f"CRITICAL ERROR starting MINIMAL server: {e}", exc_info=True)
finally:
    logger.info("Minimal server script finished.")