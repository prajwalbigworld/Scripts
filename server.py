server.py

# backend/server.py

import os
import io
import sys
import asyncio
import logging
import time
import threading
import json
import numpy as np
from typing import Dict, Optional, List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from SmartApi.smartConnect import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp
import uvicorn
import pandas as pd
import requests
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, JSONResponse
from dataclasses import dataclass
from circuitbreaker import circuit
from prometheus_client import generate_latest, start_http_server
from starlette.websockets import WebSocketState

# Local imports (models, engines, etc.)
from models.ai_core import (
    LSTMPriceDirectionModel,
    BayesianNoiseFilter,
    ReinforcementAgent,
    ModelRegistry,
    TradeAction,
    ModelInferenceError,
)
from engine.hybrid_signal_engine import HybridSignalEngine
from engine.option_chain_engine import OptionChainEngine
from core.metrics import metrics

# Additional engines and models
from engine.feature_builder import FeatureBuilder
from engine.option_signal_worker import OptionSignalWorker
from models.ai_model_validator import AIModelValidator
from engine.global_feed_engine import GlobalFeedEngine
from engine.market_regime import MarketRegimeDetector
from engine.orderflow_engine import OrderFlowEngine
from engine.option_chain_fetcher import OptionChainFetcher
from engine.rule_signal_engine import RuleSignalEngine
from engine.strike_filter import StrikeFilter
from engine.spoof_detector import SpoofDetector
from engine.iceberg_detector import IcebergDetector
from models.reinforcement_agent import RLAgentConfig
from models.lstm_model import LSTMModel
from models.model_types import AIModelType, TradeAction
from models.ai_model_server import AIModelServer
from models.model_configs import (
    LSTMModelConfig, BayesianFilterConfig, RLAgentConfig
)

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Scrip master
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
SCRIP_MASTER_CACHE = None
SCRIP_MASTER_LAST_UPDATED = None

# === SIGNAL ENGINE ===

@dataclass
class SignalOutput:
    direction: str  # BUY/SELL/NEUTRAL
    confidence: float
    rl_action: str  # EXECUTE/DELAY/CANCEL
    filtered_signal: bool
    features: Dict[str, float]

class SignalEngine:
    def __init__(self):
        self.models = self._initialize_models()
        self.feature_history = []
        self._lock = threading.Lock()
        self._ready = False
        self.orderflow_signals = []
        self.ai_predictions = []
        self.context = {}
        try:
            self._warmup()
        except Exception as e:
            logger.error(f"SignalEngine initialization failed: {str(e)}")
            self._ready = False
            raise

    def _initialize_models(self) -> ModelRegistry:
        """Initialize all AI models with proper configuration"""
        registry = ModelRegistry()
        
        lstm_config = LSTMModelConfig(
            input_size=5,
            hidden_size=128,
            num_layers=2,
            output_size=3,
            dropout=0.2,
            model_path="models/lstm_direction.h5",
            sequence_length=60,
            num_features=4,
            min_confidence=0.65
        )
        lstm = LSTMPriceDirectionModel(lstm_config)
        
        bayes_config = BayesianFilterConfig(
            prior_prob=0.5,
            likelihood_true=0.8,
            likelihood_false=0.2
        )
        bayes_filter = BayesianNoiseFilter(bayes_config)
        
        rl_config = RLAgentConfig(
            state_vars=("signal_strength", "volatility", "latency"),
            action_space=tuple(TradeAction),
            max_position_size=1000.0
        )
        rl_agent = ReinforcementAgent(rl_config)
        
        registry.register("lstm", lstm)
        registry.register("bayes_filter", bayes_filter)
        registry.register("rl_agent", rl_agent)
        
        return registry

    def _warmup(self):
        """Initialize models with production data"""
        try:
            empty_data = {
                "call_oi": 1.0, 
                "put_oi": 1.0,
                "call_iv": 0.1, 
                "put_iv": 0.1,
                "call_vol": 1.0, 
                "put_vol": 1.0,
                "price_change": 0.0, 
                "time_delta": 1.0,
                "processing_latency": 0.0
            }
            self.process_market_data(empty_data)
            self._ready = True
            logger.info("SignalEngine initialized successfully")
        except Exception as e:
            logger.error(f"SignalEngine warmup failed: {str(e)}")
            self._ready = False
            raise

    @circuit(failure_threshold=3, recovery_timeout=60)
    @metrics.get_metric("model_inference_latency").labels(model_type="lstm").time()
    def process_market_data(self, raw_data: Dict) -> Optional[SignalOutput]:
        """Main processing pipeline with circuit breaker"""
        metrics.get_metric('engine_requests').labels(engine_type='signal').inc()
        
        try:
            features = self._extract_features(raw_data)

            with self._lock:
                self.feature_history.append(features)
                if len(self.feature_history) > 1000:
                    self.feature_history.pop(0)

            lstm = self.models.get("lstm")
            lstm_probs, confidence = lstm.predict(np.array([list(features.values())]))

            bayes_filter = self.models.get("bayes_filter")
            filtered, filtered_prob = bayes_filter.update(confidence)

            rl_state = self._create_rl_state(features, lstm_probs)
            rl_agent = self.models.get("rl_agent")
            action, _ = rl_agent.decide_action(confidence, rl_state)

            return SignalOutput(
                direction=self._get_direction(lstm_probs),
                confidence=confidence,
                rl_action=action,
                filtered_signal=filtered,
                features=features
            )
        except Exception as e:
            logger.error(f"Signal processing failed: {str(e)}", exc_info=True)
            metrics.get_metric('signal_processing_errors').inc()
            raise ModelInferenceError(f"Signal processing failed: {str(e)}") from e

    def _extract_features(self, data: Dict) -> Dict[str, float]:
        """Convert raw market data into model features with numeric values"""
        def to_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        return {
            "oi_imbalance": to_float(data.get("call_oi", 1)) / (to_float(data.get("put_oi", 1)) + 1e-6),
            "iv_spread": to_float(data.get("call_iv", 0)) - to_float(data.get("put_iv", 0)),
            "volume_ratio": to_float(data.get("call_vol", 1)) / (to_float(data.get("put_vol", 1)) + 1e-6),
            "price_velocity": to_float(data.get("price_change", 0)) / (to_float(data.get("time_delta", 1)) + 1e-6),
            "processing_latency": to_float(data.get("processing_latency", 0))
        }

    def _create_rl_state(self, features: Dict, lstm_output: np.ndarray) -> np.ndarray:
        """Prepare state representation for RL agent as numpy array"""
        def safe_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        return np.array([
            safe_float(max(lstm_output)),  # signal_strength
            safe_float(features.get("iv_spread", 0)),  # volatility
            safe_float(features.get("processing_latency", 0))  # latency
        ], dtype=np.float32)

    def _get_direction(self, probabilities: np.ndarray) -> str:
        """Convert probability array to direction"""
        classes = ["BUY", "SELL", "NEUTRAL"]
        return classes[np.argmax(probabilities)]

    def update_context(self, context_data: Dict):
        """Update engine context with market regime and other data"""
        with self._lock:
            self.context.update(context_data)

    def push_orderflow_signal(self, signal: Dict):
        """Store orderflow signals for processing"""
        with self._lock:
            self.orderflow_signals.append(signal)
            if len(self.orderflow_signals) > 100:
                self.orderflow_signals.pop(0)

    def push_ai_prediction(self, prediction: Dict):
        """Store AI predictions for processing"""
        with self._lock:
            self.ai_predictions.append(prediction)
            if len(self.ai_predictions) > 100:
                self.ai_predictions.pop(0)

    def is_ready(self) -> bool:
        return self._ready

    def get_metrics(self) -> Dict:
        """Get metrics from all models"""
        metrics_data = {
            "requests": metrics.get_metric('engine_requests').labels(engine_type='signal')._value.get(),
            "processing_time": metrics.get_metric('signal_processing_time')._sum.get()
        }

        try:
            model_metrics = self.models.get_model_metrics()
            metrics_data.update(model_metrics)
        except Exception as e:
            logger.warning(f"Failed to get model metrics: {str(e)}")

        return metrics_data

#part 2

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.live_clients: Set[WebSocket] = set()
        self.signal_clients: Set[WebSocket] = set()

        self.smart_api: Optional[SmartConnect] = None
        self.session_data = None
        self.angel_ws: Optional[SmartWebSocketV2] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_ready = False
        self.loop = asyncio.get_event_loop()
        self.shutdown_flag = False

        self.subscribed_tokens: Dict[str, Dict] = {}
        self.pending_subscriptions: List = []
        self.lock = threading.Lock()

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        with self.lock:
            self.active_connections[client_id] = websocket
        logger.info(f"[Client Connected] {client_id}")

    async def disconnect(self, client_id: str):
        with self.lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                logger.info(f"[Client Disconnected] {client_id}")

    async def connect_live_client(self, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.live_clients.add(websocket)
        logger.info(f"[Live WS Client Connected] Total: {len(self.live_clients)}")

    async def broadcast_to_live(self, message: dict):
        disconnected = set()
        for ws in list(self.live_clients):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(message)
                else:
                    self.live_clients.discard(ws)
            except Exception:
                self.live_clients.discard(ws)


    def subscribe_token(self, token: str, exchangeType: int):
        token_key = f"{exchangeType}|{token}"
        if token_key in self.subscribed_tokens:
            return

        if not self.ws_ready:
            self.pending_subscriptions.append((token, exchangeType))
            return

        if SCRIP_MASTER_CACHE:
            valid = any(
                str(item["token"]) == str(token)
                and item["exch_seg"] == self.exchange_type_to_str(exchangeType)
                for item in SCRIP_MASTER_CACHE
            )
            if not valid:
                logger.warning(f"[Invalid Token] {token_key}")
                return

        try:
            self.angel_ws.subscribe(
                correlation_id=f"sub_{token}",
                mode=3,  # LTP + Market Depth
                token_list=[{
                    "exchangeType": exchangeType,
                    "tokens": [token]
                }]
            )
            self.subscribed_tokens[token_key] = {
                "exchangeType": exchangeType,
                "token": token
            }
            logger.info(f"[Subscribed] Token: {token_key}")
        except Exception as e:
            logger.error(f"[Subscription Failed] Token: {token_key} → {e}")

    def exchange_type_to_str(self, exchangeType: int) -> str:
        exchange_map = {
            1: "NSE", 2: "NFO", 3: "BSE", 4: "BFO",
            5: "MCX", 7: "CDS", 13: "NCDEX"
        }
        return exchange_map.get(exchangeType, "NSE")

    def initialize_smart_api(self) -> bool:
        try:
            self.smart_api = SmartConnect(api_key=os.getenv("API_KEY"))
            totp = pyotp.TOTP(os.getenv("TOTP_SECRET")).now()
            self.session_data = self.smart_api.generateSession(
                clientCode=os.getenv("CLIENT_CODE"),
                password=os.getenv("mpin"),
                totp=totp
            )
            if not self.session_data["status"]:
                logger.error(f"[Login Failed] {self.session_data['message']}")
                return False

            logger.info("✅ SmartAPI Authentication Successful")
            return True
        except Exception as e:
            logger.error(f"[SmartAPI Init Error] {e}")
            return False

    def start_angel_websocket(self) -> bool:
        if self.ws_thread and self.ws_thread.is_alive():
            try:
                self.angel_ws.close_connection()
                self.ws_thread.join()
            except Exception:
                pass

        if not self.initialize_smart_api():
            return False

        jwt = self.session_data["data"]["jwtToken"]
        feed_token = self.smart_api.getfeedToken()

        self.angel_ws = SmartWebSocketV2(
            auth_token=jwt,
            api_key=os.getenv("API_KEY"),
            client_code=os.getenv("CLIENT_CODE"),
            feed_token=feed_token,
            max_retry_attempt=5,
            retry_delay=3,
            retry_duration=60
        )

        def on_open(wsapp):
            logger.info("📡 WebSocket Opened")
            self.ws_ready = True
            for token, exch in self.pending_subscriptions:
                self.subscribe_token(token, exch)
            self.pending_subscriptions.clear()

        def on_data(wsapp, message):
            try:
                if isinstance(message, dict):
                    parsed = message
                else:
                    parsed = json.loads(message)
                asyncio.run_coroutine_threadsafe(self.broadcast_to_live(parsed), self.loop)
            except Exception as e:
                logger.error(f"[on_data error] {e}")


        def on_error(wsapp, error):
            logger.error(f"[WebSocket Error] {error}")
            self.ws_ready = False
            asyncio.run_coroutine_threadsafe(self.reconnect_angel_ws(), self.loop)

        def on_close(wsapp):
            logger.warning("[WebSocket Closed]")
            self.ws_ready = False
            if not self.shutdown_flag:
                asyncio.run_coroutine_threadsafe(self.reconnect_angel_ws(), self.loop)

        self.angel_ws.on_open = on_open
        self.angel_ws.on_data = on_data
        self.angel_ws.on_error = on_error
        self.angel_ws.on_close = on_close

        self.ws_thread = threading.Thread(target=self.angel_ws.connect, daemon=True)
        self.ws_thread.start()
        return True

    async def reconnect_angel_ws(self):
        logger.info("🔁 Reconnecting WebSocket...")
        self.ws_ready = False
        try:
            self.angel_ws.close_connection()
        except Exception:
            pass
        await asyncio.sleep(3)
        self.start_angel_websocket()


#part 3

app = FastAPI(title="F&O AI Trading Engine")

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting FastAPI lifespan setup")

    # Start Prometheus metrics
    start_http_server(9000)

    # --- Init Manager & Core Engines ---
    manager = ConnectionManager()
    signal_engine = SignalEngine()
    rule_engine = RuleSignalEngine()
    option_engine = OptionChainEngine()
    hybrid_engine = HybridSignalEngine(option_engine, rule_engine)
    orderflow_engine = OrderFlowEngine(window_size=60)
    regime_detector = MarketRegimeDetector()
    spoof_detector = SpoofDetector()
    iceberg_detector = IcebergDetector()
    strike_filter = StrikeFilter()
    global_feed_engine = GlobalFeedEngine()
    option_signal_worker = OptionSignalWorker()

    # --- Init AI model server ---
    try:
        ai_model_server = AIModelServer(model_type=AIModelType.LSTM)
    except Exception as e:
        logger.error(f"[AI Model Server Init Error] {e}")
        ai_model_server = None

    # --- Init AI model validator ---
    try:
        ai_model_validator = AIModelValidator(model_type=AIModelType.LSTM)
    except Exception as e:
        logger.error(f"[Model Validator Error] {e}")
        ai_model_validator = None

    # --- SmartAPI Auth + Session Init ---
    if not manager.initialize_smart_api():
        raise RuntimeError("❌ SmartAPI login failed")

    session_data = manager.session_data["data"]
    smart_api = manager.smart_api

    # --- Option Chain Fetcher ---
    symbol = os.getenv("DEFAULT_SYMBOL", "NIFTY")
    option_chain_fetcher = OptionChainFetcher(
    jwt_token=session_data["jwtToken"],
    feed_token=smart_api.getfeedToken(),
    client_code=os.getenv("CLIENT_CODE"),
    symbol=symbol
    )

    # --- App State Sharing ---
    app.state.manager = manager
    app.state.signal_engine = signal_engine
    app.state.rule_engine = rule_engine
    app.state.option_engine = option_engine
    app.state.hybrid_engine = hybrid_engine
    app.state.orderflow_engine = orderflow_engine
    app.state.market_regime_detector = regime_detector
    app.state.spoof_detector = spoof_detector
    app.state.iceberg_detector = iceberg_detector
    app.state.strike_filter = strike_filter
    app.state.global_feed_engine = global_feed_engine
    app.state.option_signal_worker = option_signal_worker
    app.state.option_chain_fetcher = option_chain_fetcher
    app.state.ai_model_server = ai_model_server
    app.state.ai_model_validator = ai_model_validator

    # --- Start WebSocket Feed ---
    if not manager.start_angel_websocket():
        logger.error("❌ WebSocket start failed")
        raise RuntimeError("WebSocket initialization failed")

    # --- Start background tasks ---
    if option_chain_fetcher:
        asyncio.create_task(
            option_chain_fetcher.run_forever(callback=handle_option_chain_tick)
        )

    if option_signal_worker:
        asyncio.create_task(
            option_signal_worker.run_forever(signal_engine, manager.signal_clients)
        )

    app.state.healthy = all([
        manager.ws_ready,
        signal_engine.is_ready(),
        ai_model_server is not None,
        ai_model_validator is not None
    ])

    logger.info("✅ Lifespan startup complete")
    yield  # === APP STARTS HERE ===

    # === Shutdown Sequence ===
    logger.info("🛑 Lifespan shutdown initiated")
    manager.shutdown_flag = True

    if manager.angel_ws:
        try:
            manager.angel_ws.close_connection()
        except:
            pass

    if manager.ws_thread and manager.ws_thread.is_alive():
        manager.ws_thread.join(timeout=3)

    await option_chain_fetcher.stop()
    await option_signal_worker.shutdown()

app.router.lifespan_context = lifespan

#part 4

@app.get("/health")
async def health_check():
    return {
        "websocket_connected": app.state.manager.ws_ready,
        "signal_engine_ready": app.state.signal_engine.is_ready(),
        "healthy": app.state.healthy
    }

@app.get("/metrics")
async def get_metrics():
    return Response(
        content=generate_latest(registry=metrics.registry),
        media_type="text/plain"
    )

@app.post("/predict")
async def predict(payload: dict):
    try:
        prediction = app.state.ai_model_server.predict(payload["features"])
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate(payload: dict):
    try:
        result = app.state.ai_model_validator.evaluate(payload["X"], payload["y"])
        return result
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    client_id = f"client_{len(app.state.manager.active_connections) + 1}"
    await app.state.manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text("pong")
                elif msg.get("type") == "subscribe":
                    tokens = msg.get("tokens", [])
                    exch = msg.get("exchangeType", 2)
                    for token in tokens:
                        app.state.manager.subscribe_token(token, exch)
                    await websocket.send_json({"status": "subscribed", "tokens": tokens})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        await app.state.manager.disconnect(client_id)

@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    await app.state.manager.connect_live_client(websocket)
    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        app.state.manager.live_clients.discard(websocket)

async def handle_option_chain_tick(chain_data: Dict):
    manager = app.state.manager
    top_strikes = app.state.strike_filter.filter(chain_data)
    spoof_flags = app.state.spoof_detector.detect(chain_data)
    iceberg_flags = app.state.iceberg_detector.detect(chain_data)
    option_metrics = app.state.option_engine.compute_metrics(top_strikes)
    regime = app.state.market_regime_detector.detect()

    app.state.signal_engine.update_context({
        "regime": regime,
        "option_metrics": option_metrics,
        "spoof_flags": spoof_flags,
        "iceberg_flags": iceberg_flags
    })

    ai_features = option_metrics.get("features", [])
    prediction = app.state.ai_model_server.predict(ai_features)

    tick = chain_data.get("trade_tick")
    if tick:
        app.state.orderflow_engine.update(tick)
        signal = app.state.orderflow_engine.compute_pressure()
        if signal.get("status") in ["bullish", "bearish"]:
            app.state.signal_engine.push_orderflow_signal(signal)

    app.state.signal_engine.push_ai_prediction(prediction)

    live_data = {
        "strikes": top_strikes,
        "prediction": prediction,
        "regime": regime,
        "spoof": spoof_flags,
        "iceberg": iceberg_flags
    }
    await manager.broadcast_to_live(live_data)

if __name__ == "__main__":
    print("🚀 Launching F&O AI Trading Backend...")
    uvicorn.run(
        "backend.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        ws_ping_interval=15,
        ws_ping_timeout=30,
        log_level="info"
    )
