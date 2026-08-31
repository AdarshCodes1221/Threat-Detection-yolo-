import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  CircleHelp,
  Clock3,
  Cpu,
  Gauge,
  LayoutDashboard,
  Monitor,
  Radio,
  RefreshCw,
  Settings,
  Shield,
  ShieldCheck,
  Users,
  Video,
  VideoOff,
  Wifi,
  WifiOff,
  XCircle,
} from "lucide-react";

import "./App.css";


const API_URL =
  "http://127.0.0.1:8000";

const WS_URL =
  "ws://127.0.0.1:8000/ws/detect";


const DEFAULT_STATS = {
  persons: 0,
  threats: 0,
  objects: 0,
  highestConfidence: 0,
  processingMs: 0,
  fps: 0,
};


const PAGE_META = {
  monitor: {
    title: "Live Monitor",
    description:
      "Real-time camera surveillance and detection",
  },

  events: {
    title: "Detection Events",
    description:
      "Tracked entities observed during this session",
  },

  analytics: {
    title: "Analytics",
    description:
      "Live performance and detection statistics",
  },

  settings: {
    title: "System Settings",
    description:
      "Detection and application configuration",
  },

  about: {
    title: "System Information",
    description:
      "Threat Detection YOLO11 platform information",
  },
};


// ============================================================
// APP
// ============================================================

function App() {

  // ==========================================================
  // CAMERA
  // ==========================================================

  const videoRef =
    useRef(null);

  const canvasRef =
    useRef(null);


  // ==========================================================
  // NETWORK
  // ==========================================================

  const streamRef =
    useRef(null);

  const socketRef =
    useRef(null);

  const reconnectTimerRef =
    useRef(null);


  // ==========================================================
  // CONNECTION FLAGS
  // ==========================================================

  const shouldReconnectRef =
    useRef(false);

  const frameInFlightRef =
    useRef(false);


  // ==========================================================
  // EVENT MAP
  // ==========================================================

  const eventMapRef =
    useRef(new Map());


  // ==========================================================
  // PAGE
  // ==========================================================

  const [
    activePage,
    setActivePage,
  ] = useState("monitor");


  // ==========================================================
  // BACKEND
  // ==========================================================

  const [
    backendConnected,
    setBackendConnected,
  ] = useState(false);


  // ==========================================================
  // CAMERA
  // ==========================================================

  const [
    cameraRunning,
    setCameraRunning,
  ] = useState(false);


  // ==========================================================
  // SOCKET
  // ==========================================================

  const [
    socketConnected,
    setSocketConnected,
  ] = useState(false);


  // ==========================================================
  // PROCESSED FRAME
  //
  // THIS FIXES THE BLACK SCREEN.
  // ==========================================================

  const [
    processedFrame,
    setProcessedFrame,
  ] = useState("");


  // ==========================================================
  // ERROR
  // ==========================================================

  const [
    cameraError,
    setCameraError,
  ] = useState("");


  // ==========================================================
  // SETTINGS
  // ==========================================================

  const [
    confidence,
    setConfidence,
  ] = useState(0.35);


  const [
    inferenceWidth,
    setInferenceWidth,
  ] = useState(480);


  const [
    showTrackIds,
    setShowTrackIds,
  ] = useState(true);


  const [
    saveEvents,
    setSaveEvents,
  ] = useState(false);


  // ==========================================================
  // DETECTION
  // ==========================================================

  const [
    stats,
    setStats,
  ] = useState(
    DEFAULT_STATS
  );


  const [
    events,
    setEvents,
  ] = useState([]);


  // ==========================================================
  // MODEL
  // ==========================================================

  const [
    modelInfo,
    setModelInfo,
  ] = useState({
    name: "--",
    classes: 0,
    device: "--",
    deviceName: "--",
  });


  // ==========================================================
  // SESSION
  // ==========================================================

  const [
    session,
    setSession,
  ] = useState({
    frames: 0,
    personObservations: 0,
    threatObservations: 0,
    confidenceSum: 0,
    confidenceSamples: 0,
    totalProcessingMs: 0,
  });


  // ==========================================================
  // CLOCK
  // ==========================================================

  const [
    clock,
    setClock,
  ] = useState(
    new Date()
  );


  // ==========================================================
  // CLOCK
  // ==========================================================

  useEffect(() => {

    const timer =
      window.setInterval(
        () => {
          setClock(
            new Date()
          );
        },
        1000
      );

    return () => {
      window.clearInterval(
        timer
      );
    };

  }, []);


  // ==========================================================
  // BACKEND HEALTH
  // ==========================================================

  const checkBackend =
    useCallback(
      async () => {

        try {

          const response =
            await fetch(
              `${API_URL}/api/health`,
              {
                cache:
                  "no-store",
              }
            );


          if (!response.ok) {
            throw new Error(
              "Backend unavailable"
            );
          }


          const data =
            await response.json();


          setBackendConnected(
            true
          );


          setModelInfo({
            name:
              data.model_name ||
              "--",

            classes:
              Number(
                data.class_count ||
                0
              ),

            device:
              data.device === 0
                ? "CUDA"
                : String(
                    data.device ||
                    "--"
                  ),

            deviceName:
              data.device_name ||
              "--",
          });


        } catch {

          setBackendConnected(
            false
          );

        }

      },
      []
    );


  useEffect(() => {

    checkBackend();

    const timer =
      window.setInterval(
        checkBackend,
        5000
      );

    return () => {
      window.clearInterval(
        timer
      );
    };

  }, [
    checkBackend,
  ]);


  // ==========================================================
  // CONFIDENCE UPDATE
  // ==========================================================

  useEffect(() => {

    if (!backendConnected) {
      return;
    }


    fetch(
      `${API_URL}/api/confidence`,
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify({
          value: confidence,
        }),
      }
    ).catch(() => {});

  }, [
    confidence,
    backendConnected,
  ]);


  // ==========================================================
  // EVENT UPDATE
  // ==========================================================

  const updateEvents =
    useCallback(
      (detections) => {

        if (
          !Array.isArray(
            detections
          )
        ) {
          return;
        }


        const now =
          new Date();

        const updates = [];


        for (
          const detection
          of detections
        ) {

          const trackId =
            detection.track_id;


          if (
            trackId ===
              null ||
            trackId ===
              undefined
          ) {
            continue;
          }


          const className =
            detection.class_name ||
            "unknown";


          const key =
            `${trackId}:${className}`;


          const oldEvent =
            eventMapRef.current.get(
              key
            );


          const personId = detection.person_id || detection.associated_person_id || null;
          const weaponId = detection.weapon_id || null;
          const phoneId = detection.phone_id || null;
          const associatedPersonId = detection.associated_person_id || personId || null;

          const event = {

            id:
              oldEvent?.id ||
              `${key}-${Date.now()}`,

            trackId,

            timestamp:
              now.getTime(),

            time:
              now.toLocaleTimeString(
                [],
                {
                  hour:
                    "2-digit",

                  minute:
                    "2-digit",

                  second:
                    "2-digit",
                }
              ),

            type:
              detection.is_threat
                ? `THREAT EVENT ${detection.event_id || ""}`.trim()
                : detection.is_person
                  ? `P-${String((personId || trackId || "")).replace(/[^0-9]/g, "") || "??"}`
                  : className === "weapon"
                    ? `W-${String((weaponId || trackId || "")).replace(/[^0-9]/g, "") || "??"}`
                    : className === "phone"
                      ? `PH-${String((phoneId || trackId || "")).replace(/[^0-9]/g, "") || "??"}`
                      : className,

            className,
            personId,
            weaponId,
            phoneId,
            associatedPersonId,

            confidence:
              Number(
                detection.confidence ||
                0
              ),

            isThreat:
              Boolean(
                detection.is_threat
              ),

            isPerson:
              Boolean(
                detection.is_person
              ),
          };


          eventMapRef.current.set(
            key,
            event
          );


          updates.push(
            event
          );
        }


        if (
          updates.length ===
          0
        ) {
          return;
        }


        setEvents(
          (current) => {

            const merged =
              new Map(
                current.map(
                  (event) => [
                    event.id,
                    event,
                  ]
                )
              );


            updates.forEach(
              (event) => {
                merged.set(
                  event.id,
                  event
                );
              }
            );


            return Array
              .from(
                merged.values()
              )
              .sort(
                (a, b) =>
                  b.timestamp -
                  a.timestamp
              )
              .slice(
                0,
                100
              );

          }
        );

      },
      []
    );


  // ==========================================================
  // HANDLE DETECTION
  // ==========================================================

  const handleDetection =
    useCallback(
      (data) => {

        // Previous request completed.
        frameInFlightRef.current =
          false;


        if (
          data?.type ===
          "error"
        ) {

          console.warn(
            "[Backend]",
            data.message
          );

          return;
        }


        if (
          data?.type !==
          "frame"
        ) {
          return;
        }


        // ----------------------------------------------------
        // CRITICAL FIX
        //
        // Store returned image in React state.
        // Don't depend on a conditionally-rendered img ref.
        // ----------------------------------------------------

        if (
          data.frame
        ) {

          setProcessedFrame(
            data.frame
          );

        }


        // ----------------------------------------------------
        // STATS
        // ----------------------------------------------------

        const processingMs =
          Number(
            data.processing_ms ||
            0
          );


        setStats({

          persons:
            Number(
              data.persons ||
              0
            ),

          threats:
            Number(
              data.threats ||
              0
            ),

          objects:
            Number(
              data.objects ||
              0
            ),

          highestConfidence:
            Number(
              data.highest_confidence ||
              0
            ),

          processingMs,

          fps:
            processingMs > 0
              ? 1000 /
                processingMs
              : 0,
        });


        // ----------------------------------------------------
        // MODEL
        // ----------------------------------------------------

        if (
          data.model
        ) {

          setModelInfo(
            (previous) => ({
              ...previous,
              name:
                data.model,
            })
          );

        }


        // ----------------------------------------------------
        // SESSION
        // ----------------------------------------------------

        setSession(
          (previous) => ({

            ...previous,

            frames:
              previous.frames +
              1,

            personObservations:
              previous.personObservations +
              Number(
                data.persons ||
                0
              ),

            threatObservations:
              previous.threatObservations +
              Number(
                data.threats ||
                0
              ),

            confidenceSum:
              previous.confidenceSum +
              Number(
                data.highest_confidence ||
                0
              ),

            confidenceSamples:
              previous.confidenceSamples +
              (
                data.highest_confidence >
                0
                  ? 1
                  : 0
              ),

            totalProcessingMs:
              previous.totalProcessingMs +
              processingMs,

          })
        );


        // ----------------------------------------------------
        // EVENTS
        // ----------------------------------------------------

        updateEvents(
          data.detections
        );

      },
      [
        updateEvents,
      ]
    );


  // ==========================================================
  // SEND FRAME
  // ==========================================================

  const sendFrame =
    useCallback(
      () => {

        const video =
          videoRef.current;

        const canvas =
          canvasRef.current;

        const socket =
          socketRef.current;


        if (
          !video ||
          !canvas ||
          !socket
        ) {
          return;
        }


        if (
          socket.readyState !==
          WebSocket.OPEN
        ) {
          return;
        }


        // ----------------------------------------------------
        // BACK-PRESSURE
        //
        // Never send frame 2 until frame 1 has returned.
        // ----------------------------------------------------

        if (
          frameInFlightRef.current
        ) {
          return;
        }


        if (
          video.videoWidth <= 0 ||
          video.videoHeight <= 0
        ) {
          return;
        }


        frameInFlightRef.current =
          true;


        const width =
          inferenceWidth;


        const aspectRatio =
          video.videoHeight /
          video.videoWidth;


        const height =
          Math.round(
            width *
              aspectRatio
          );


        canvas.width =
          width;

        canvas.height =
          height;


        const context =
          canvas.getContext(
            "2d",
            {
              alpha: false,
            }
          );


        if (!context) {

          frameInFlightRef.current =
            false;

          return;
        }


        context.drawImage(
          video,
          0,
          0,
          width,
          height
        );


        const frame =
          canvas.toDataURL(
            "image/jpeg",
            0.55
          );


        try {

          socket.send(
            JSON.stringify({
              frame,
            })
          );


          console.log(
            "[Camera] Frame sent"
          );

        } catch (
          error
        ) {

          console.error(
            "[Camera] Send failed:",
            error
          );

          frameInFlightRef.current =
            false;

        }

      },
      [
        inferenceWidth,
      ]
    );


  // ==========================================================
  // CONNECT SOCKET
  // ==========================================================

  const connectSocket =
    useCallback(
      () => {

        if (
          !shouldReconnectRef.current
        ) {
          return;
        }


        const existing =
          socketRef.current;


        if (
          existing &&
          (
            existing.readyState ===
              WebSocket.OPEN ||
            existing.readyState ===
              WebSocket.CONNECTING
          )
        ) {
          return;
        }


        console.log(
          "[WS] Connecting..."
        );


        const socket =
          new WebSocket(
            WS_URL
          );


        socketRef.current =
          socket;


        socket.onopen =
          () => {

            console.log(
              "[WS] Connected"
            );


            setSocketConnected(
              true
            );


            frameInFlightRef.current =
              false;


            // Wait briefly for the
            // camera frame to stabilize.
            window.setTimeout(
              () => {

                if (
                  shouldReconnectRef.current
                ) {
                  sendFrame();
                }

              },
              200
            );

          };


        socket.onmessage =
          (message) => {

            try {

              const data =
                JSON.parse(
                  message.data
                );


              console.log(
                "[WS] Result:",
                data.type
              );


              handleDetection(
                data
              );


              // ------------------------------------------------
              // Send next frame only after
              // receiving the previous result.
              // ------------------------------------------------

              if (
                shouldReconnectRef.current &&
                socket.readyState ===
                  WebSocket.OPEN
              ) {

                window.requestAnimationFrame(
                  () => {
                    sendFrame();
                  }
                );

              }

            } catch (
              error
            ) {

              console.error(
                "[WS] Response error:",
                error
              );


              frameInFlightRef.current =
                false;

            }

          };


        socket.onerror =
          (error) => {

            console.error(
              "[WS] Error:",
              error
            );


            setSocketConnected(
              false
            );


            frameInFlightRef.current =
              false;

          };


        socket.onclose =
          () => {

            console.log(
              "[WS] Closed"
            );


            setSocketConnected(
              false
            );


            frameInFlightRef.current =
              false;


            socketRef.current =
              null;


            if (
              shouldReconnectRef.current
            ) {

              reconnectTimerRef.current =
                window.setTimeout(
                  () => {
                    connectSocket();
                  },
                  1000
                );

            }

          };

      },
      [
        handleDetection,
        sendFrame,
      ]
    );


  // ==========================================================
  // START CAMERA
  // ==========================================================

  const startCamera =
    useCallback(
      async () => {

        setCameraError("");


        if (
          !backendConnected
        ) {

          setCameraError(
            "Detection backend is offline."
          );

          return;
        }


        try {

          // --------------------------------------------------
          // OPEN CAMERA
          // --------------------------------------------------

          if (
            !streamRef.current
          ) {

            const stream =
              await navigator
                .mediaDevices
                .getUserMedia(
                  {
                    video: {
                      width: {
                        ideal:
                          1280,
                      },

                      height: {
                        ideal:
                          720,
                      },

                      frameRate: {
                        ideal:
                          24,
                      },
                    },

                    audio:
                      false,
                  }
                );


            streamRef.current =
              stream;

          }


          const video =
            videoRef.current;


          if (!video) {

            throw new Error(
              "Camera element is not ready."
            );

          }


          video.srcObject =
            streamRef.current;


          video.muted =
            true;


          video.playsInline =
            true;


          // --------------------------------------------------
          // WAIT FOR CAMERA METADATA
          // --------------------------------------------------

          if (
            video.readyState <
            HTMLMediaElement.HAVE_METADATA
          ) {

            await new Promise(
              (resolve) => {

                const handler =
                  () => {

                    video.removeEventListener(
                      "loadedmetadata",
                      handler
                    );

                    resolve();
                  };


                video.addEventListener(
                  "loadedmetadata",
                  handler
                );

              }
            );

          }


          await video.play();


          console.log(
            "[Camera] Started:",
            video.videoWidth,
            "x",
            video.videoHeight
          );


          setProcessedFrame(
            ""
          );


          frameInFlightRef.current =
            false;


          shouldReconnectRef.current =
            true;


          setCameraRunning(
            true
          );


          connectSocket();

        } catch (
          error
        ) {

          console.error(
            "[Camera] Start error:",
            error
          );


          shouldReconnectRef.current =
            false;


          setCameraRunning(
            false
          );


          setCameraError(
            error?.message ||
              "Unable to access webcam."
          );

        }

      },
      [
        backendConnected,
        connectSocket,
      ]
    );


  // ==========================================================
  // STOP CAMERA
  // ==========================================================

  const stopCamera =
    useCallback(
      () => {

        shouldReconnectRef.current =
          false;


        frameInFlightRef.current =
          false;


        if (
          reconnectTimerRef.current
        ) {

          window.clearTimeout(
            reconnectTimerRef.current
          );

          reconnectTimerRef.current =
            null;

        }


        if (
          socketRef.current
        ) {

          socketRef.current.close();

          socketRef.current =
            null;

        }


        if (
          streamRef.current
        ) {

          streamRef.current
            .getTracks()
            .forEach(
              (
                track
              ) => {
                track.stop();
              }
            );


          streamRef.current =
            null;

        }


        if (
          videoRef.current
        ) {

          videoRef.current.pause();

          videoRef.current.srcObject =
            null;

        }


        setSocketConnected(
          false
        );


        setCameraRunning(
          false
        );


        setProcessedFrame(
          ""
        );


        setStats(
          DEFAULT_STATS
        );

      },
      []
    );


  // ==========================================================
  // CLEANUP
  // ==========================================================

  useEffect(() => {

    return () => {
      stopCamera();
    };

  }, [
    stopCamera,
  ]);


  // ==========================================================
  // DERIVED DATA
  // ==========================================================

  const averageConfidence =
    session.confidenceSamples >
    0
      ? session.confidenceSum /
        session.confidenceSamples
      : 0;


  const averageProcessing =
    session.frames >
    0
      ? session.totalProcessingMs /
        session.frames
      : 0;


  const currentStatus =
    stats.threats >
    0
      ? "REVIEW REQUIRED"
      : cameraRunning
        ? "MONITORING"
        : "READY";


  const timeText =
    clock.toLocaleTimeString(
      [],
      {
        hour:
          "2-digit",

        minute:
          "2-digit",

        second:
          "2-digit",
      }
    );


  const dateText =
    clock.toLocaleDateString(
      [],
      {
        weekday:
          "short",

        month:
          "short",

        day:
          "numeric",

        year:
          "numeric",
      }
    );


  const page =
    PAGE_META[
      activePage
    ];


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app-shell">

      {/* ====================================================
          SIDEBAR
          ==================================================== */}

      <aside className="sidebar">

        <div className="sidebar-brand">

          <div className="brand-mark">

            <ShieldCheck
              size={21}
            />

          </div>


          <div>

            <div className="brand-name">
              THREAT DETECTION
            </div>

            <div className="brand-product">
              CONTROL CENTER
            </div>

          </div>

        </div>


        <div className="sidebar-section-label">
          OPERATIONS
        </div>


        <nav className="navigation">

          <NavigationItem
            icon={
              <LayoutDashboard
                size={17}
              />
            }
            label="Live Monitor"
            active={
              activePage ===
              "monitor"
            }
            onClick={() =>
              setActivePage(
                "monitor"
              )
            }
          />


          <NavigationItem
            icon={
              <Bell
                size={17}
              />
            }
            label="Detection Events"
            badge={
              events.length >
              0
                ? events.length
                : null
            }
            active={
              activePage ===
              "events"
            }
            onClick={() =>
              setActivePage(
                "events"
              )
            }
          />


          <NavigationItem
            icon={
              <BarChart3
                size={17}
              />
            }
            label="Analytics"
            active={
              activePage ===
              "analytics"
            }
            onClick={() =>
              setActivePage(
                "analytics"
              )
            }
          />


          <div className="sidebar-section-label secondary">
            SYSTEM
          </div>


          <NavigationItem
            icon={
              <Settings
                size={17}
              />
            }
            label="Settings"
            active={
              activePage ===
              "settings"
            }
            onClick={() =>
              setActivePage(
                "settings"
              )
            }
          />


          <NavigationItem
            icon={
              <CircleHelp
                size={17}
              />
            }
            label="System Info"
            active={
              activePage ===
              "about"
            }
            onClick={() =>
              setActivePage(
                "about"
              )
            }
          />

        </nav>


        <div className="sidebar-spacer" />


        <div className="sidebar-status">

          <div className="status-heading">
            SYSTEM STATUS
          </div>


          <StatusRow
            label="Backend"
            value={
              backendConnected
                ? "Connected"
                : "Offline"
            }
            state={
              backendConnected
                ? "good"
                : "bad"
            }
          />


          <StatusRow
            label="Detection"
            value={
              socketConnected
                ? "Streaming"
                : cameraRunning
                  ? "Reconnecting"
                  : "Idle"
            }
            state={
              socketConnected
                ? "good"
                : cameraRunning
                  ? "warn"
                  : "neutral"
            }
          />


          <StatusRow
            label="Model"
            value={
              modelInfo.name
            }
          />


          <StatusRow
            label="Device"
            value={
              modelInfo.device
            }
          />

        </div>


        <div className="sidebar-footer">
          Threat Detection YOLO11
        </div>

      </aside>


      {/* ====================================================
          MAIN
          ==================================================== */}

      <main className="main-content">

        <header className="main-header">

          <div>

            <div className="breadcrumb">

              SECURITY OPERATIONS

              <span>
                /
              </span>

              {page.title}

            </div>


            <div className="header-title-row">

              <h1>
                {page.title}
              </h1>


              {cameraRunning && (

                <div className="header-live">

                  <span />

                  LIVE

                </div>

              )}

            </div>


            <p className="header-description">
              {page.description}
            </p>

          </div>


          <div className="header-actions">

            <div className="system-clock">

              <Clock3
                size={14}
              />

              <div>

                <strong>
                  {timeText}
                </strong>

                <span>
                  {dateText}
                </span>

              </div>

            </div>


            {activePage ===
              "monitor" && (

              <button
                type="button"
                className={
                  cameraRunning
                    ? "primary-action danger"
                    : "primary-action"
                }
                onClick={
                  cameraRunning
                    ? stopCamera
                    : startCamera
                }
                disabled={
                  !backendConnected
                }
              >

                {cameraRunning ? (
                  <>

                    <VideoOff
                      size={16}
                    />

                    Stop Camera

                  </>
                ) : (
                  <>

                    <Video
                      size={16}
                    />

                    Start Camera

                  </>
                )}

              </button>

            )}

          </div>

        </header>


        {/* ==================================================
            LIVE MONITOR
            ================================================== */}

        <div
          className={
            activePage ===
            "monitor"
              ? "page-visible"
              : "page-hidden"
          }
        >

          <section className="monitor-layout">

            <div className="camera-column">

              <div className="camera-panel panel">

                <div className="panel-header">

                  <div>

                    <div className="panel-title-row">

                      <h2>
                        Live Camera
                      </h2>


                      <span
                        className={
                          socketConnected
                            ? "feed-status live"
                            : "feed-status"
                        }
                      >

                        {socketConnected
                          ? "STREAMING"
                          : cameraRunning
                            ? "CONNECTING"
                            : "OFFLINE"}

                      </span>

                    </div>


                    <p>
                      Browser camera →
                      YOLO11 + ByteTrack
                    </p>

                  </div>


                  <div className="camera-meta">

                    <Monitor
                      size={14}
                    />

                    {inferenceWidth}
                    px

                  </div>

                </div>


                <div className="camera-frame">

                  {/* Actual browser camera source */}
                  <video
                    ref={videoRef}
                    className="camera-input"
                    muted
                    autoPlay
                    playsInline
                  />


                  {/* Actual backend processed frame */}
                  {processedFrame ? (

                    <img
                      src={
                        processedFrame
                      }
                      className="processed-frame visible"
                      alt="Processed camera feed"
                    />

                  ) : (

                    <div className="camera-placeholder">

                      {!cameraRunning ? (
                        <>

                          <div className="offline-icon">

                            <VideoOff
                              size={28}
                            />

                          </div>


                          <strong>
                            Camera Offline
                          </strong>


                          <span>
                            Start the camera
                            to begin monitoring.
                          </span>

                        </>
                      ) : socketConnected ? (
                        <>

                          <RefreshCw
                            size={22}
                            className="spin"
                          />


                          <strong>
                            Processing Camera
                          </strong>


                          <span>
                            Waiting for the
                            first processed frame.
                          </span>

                        </>
                      ) : (
                        <>

                          <RefreshCw
                            size={22}
                            className="spin"
                          />


                          <strong>
                            Connecting
                          </strong>


                          <span>
                            Connecting to
                            detection service.
                          </span>

                        </>
                      )}

                    </div>

                  )}


                  {processedFrame &&
                    socketConnected &&
                    stats.threats >
                      0 && (

                    <div className="threat-banner">

                      <AlertTriangle
                        size={15}
                      />

                      Potential threat detected

                    </div>

                  )}

                </div>


                <canvas
                  ref={canvasRef}
                  className="hidden-canvas"
                />


                <div className="camera-footer">

                  <div className="feed-source">

                    <Radio
                      size={13}
                    />

                    Integrated Webcam

                  </div>


                  <div>
                    1280 × 720 source
                  </div>

                </div>


                {cameraError && (

                  <div className="error-banner">

                    <XCircle
                      size={14}
                    />

                    {cameraError}

                  </div>

                )}

              </div>


              {/* THRESHOLD IS DIRECTLY UNDER CAMERA */}

              <section className="panel detection-control">

                <div>

                  <div className="panel-title">
                    Detection Threshold
                  </div>


                  <p>
                    Minimum confidence required
                    for a returned detection.
                  </p>

                </div>


                <div className="threshold-control">

                  <strong>
                    {confidence.toFixed(
                      2
                    )}
                  </strong>


                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.05"
                    value={
                      confidence
                    }
                    onChange={(
                      event
                    ) =>
                      setConfidence(
                        Number(
                          event.target.value
                        )
                      )
                    }
                  />

                </div>

              </section>

            </div>


            {/* ==================================================
                RIGHT COLUMN
                ================================================== */}

            <aside className="monitor-side">

              <section className="panel">

                <div className="panel-heading">

                  <div>

                    <h2>
                      Current Status
                    </h2>

                    <p>
                      Live detection state
                    </p>

                  </div>


                  <div
                    className={
                      stats.threats >
                      0
                        ? "state-indicator danger"
                        : cameraRunning
                          ? "state-indicator good"
                          : "state-indicator neutral"
                    }
                  >

                    {currentStatus}

                  </div>

                </div>


                <div className="large-status">

                  <div
                    className={
                      stats.threats >
                      0
                        ? "status-symbol danger"
                        : "status-symbol good"
                    }
                  >

                    {stats.threats >
                    0 ? (
                      <AlertTriangle
                        size={23}
                      />
                    ) : (
                      <ShieldCheck
                        size={23}
                      />
                    )}

                  </div>


                  <div>

                    <strong>

                      {stats.threats >
                      0
                        ? "Potential threat detected"
                        : cameraRunning
                          ? "System monitoring"
                          : "System ready"}

                    </strong>


                    <span>

                      {stats.threats >
                      0
                        ? "Human review recommended"
                        : cameraRunning
                          ? "Detection engine active"
                          : "Start the camera to begin"}

                    </span>

                  </div>

                </div>

              </section>


              <section className="metric-grid">

                <MetricCard
                  icon={
                    <Users
                      size={18}
                    />
                  }
                  title="Persons"
                  value={
                    stats.persons
                  }
                  detail="Current frame"
                  type="blue"
                />


                <MetricCard
                  icon={
                    <AlertTriangle
                      size={18}
                    />
                  }
                  title="Threats"
                  value={
                    stats.threats
                  }
                  detail="Current frame"
                  type="red"
                />


                <MetricCard
                  icon={
                    <Gauge
                      size={18}
                    />
                  }
                  title="Confidence"
                  value={
                    stats.highestConfidence >
                    0
                      ? `${(
                          stats.highestConfidence *
                          100
                        ).toFixed(
                          0
                        )}%`
                      : "--"
                  }
                  detail="Highest detection"
                  type="green"
                />


                <MetricCard
                  icon={
                    <Activity
                      size={18}
                    />
                  }
                  title="Inference"
                  value={
                    stats.processingMs >
                    0
                      ? `${stats.processingMs.toFixed(
                          0
                        )} ms`
                      : "--"
                  }
                  detail={
                    stats.fps >
                    0
                      ? `${stats.fps.toFixed(
                          1
                        )} FPS`
                      : "Waiting"
                  }
                  type="amber"
                />

              </section>


              <section className="panel compact-events">

                <div className="panel-heading">

                  <div>

                    <h2>
                      Recent Detections
                    </h2>

                    <p>
                      Tracked entities
                    </p>

                  </div>


                  <button
                    type="button"
                    className="text-button"
                    onClick={() =>
                      setActivePage(
                        "events"
                      )
                    }
                  >
                    View all
                  </button>

                </div>


                <div className="event-list">

                  {events
                    .slice(
                      0,
                      6
                    )
                    .map(
                      (event) => (

                        <EventRow
                          key={
                            event.id
                          }
                          event={
                            event
                          }
                          showTrackIds={
                            showTrackIds
                          }
                        />

                      )
                    )}


                  {events.length ===
                    0 && (

                    <div className="empty-events">
                      No detections recorded
                    </div>

                  )}

                </div>

              </section>

            </aside>

          </section>

        </div>


        {/* ==================================================
            EVENTS
            ================================================== */}

        <div
          className={
            activePage ===
            "events"
              ? "page-visible"
              : "page-hidden"
          }
        >

          <section className="page-section">

            <div className="events-toolbar">

              <div>

                <span className="section-kicker">
                  SESSION EVENTS
                </span>

                <h2>
                  Detection History
                </h2>

                <p>
                  Latest state for each
                  tracked entity/class pair.
                </p>

              </div>


              <div className="event-summary">

                <div>

                  <span>
                    RECORDS
                  </span>

                  <strong>
                    {events.length}
                  </strong>

                </div>


                <div>

                  <span>
                    PERSON OBS.
                  </span>

                  <strong>
                    {
                      session.personObservations
                    }
                  </strong>

                </div>


                <div>

                  <span>
                    THREAT OBS.
                  </span>

                  <strong className="danger-text">
                    {
                      session.threatObservations
                    }
                  </strong>

                </div>

              </div>

            </div>


            <div className="panel full-events-panel">

              {events.length ===
                0 ? (

                <div className="large-empty">

                  <Bell
                    size={30}
                  />

                  <strong>
                    No detection events
                  </strong>

                  <span>
                    Start the camera to
                    populate this page.
                  </span>

                </div>

              ) : (

                <div className="full-event-table">

                  <div className="table-header">

                    <span>
                      TIME
                    </span>

                    <span>
                      TRACK
                    </span>

                    <span>
                      TYPE
                    </span>

                    <span>
                      CLASS
                    </span>

                    <span>
                      CONFIDENCE
                    </span>

                  </div>


                  {events.map(
                    (event) => (

                      <div
                        className="table-row"
                        key={
                          event.id
                        }
                      >

                        <span>
                          {event.time}
                        </span>


                        <span>
                          {showTrackIds
                            ? `P-${String(
                                event.trackId
                              ).padStart(
                                2,
                                "0"
                              )}`
                            : "—"}
                        </span>


                        <span
                          className={
                            event.isThreat
                              ? "danger-text"
                              : ""
                          }
                        >
                          {event.type}
                        </span>


                        <span className="muted-cell">
                          {event.className}
                        </span>


                        <span className="confidence-cell">
                          {(
                            event.confidence *
                            100
                          ).toFixed(
                            1
                          )}
                          %
                        </span>

                      </div>

                    )
                  )}

                </div>

              )}

            </div>

          </section>

        </div>


        {/* ==================================================
            ANALYTICS
            ================================================== */}

        <div
          className={
            activePage ===
            "analytics"
              ? "page-visible"
              : "page-hidden"
          }
        >

          <section className="page-section">

            <div className="analytics-summary">

              <MetricCard
                icon={
                  <Activity
                    size={18}
                  />
                }
                title="Frames Processed"
                value={
                  session.frames
                }
                detail="Current session"
                type="blue"
              />


              <MetricCard
                icon={
                  <Users
                    size={18}
                  />
                }
                title="Tracked Records"
                value={
                  events.length
                }
                detail="Current session"
                type="green"
              />


              <MetricCard
                icon={
                  <Shield
                    size={18}
                  />
                }
                title="Threat Observations"
                value={
                  session.threatObservations
                }
                detail="Across processed frames"
                type="red"
              />


              <MetricCard
                icon={
                  <Gauge
                    size={18}
                  />
                }
                title="Average Confidence"
                value={
                  averageConfidence >
                  0
                    ? `${(
                        averageConfidence *
                        100
                      ).toFixed(
                        1
                      )}%`
                    : "--"
                }
                detail="Session average"
                type="amber"
              />

            </div>


            <div className="analytics-columns">

              <div className="panel">

                <div className="panel-heading">

                  <div>

                    <h2>
                      Processing Performance
                    </h2>

                    <p>
                      Runtime characteristics
                    </p>

                  </div>

                </div>


                <div className="analytics-list">

                  <AnalyticsRow
                    label="Current FPS"
                    value={
                      stats.fps >
                      0
                        ? stats.fps.toFixed(
                            1
                          )
                        : "--"
                    }
                  />


                  <AnalyticsRow
                    label="Average inference"
                    value={
                      averageProcessing >
                      0
                        ? `${averageProcessing.toFixed(
                            1
                          )} ms`
                        : "--"
                    }
                  />


                  <AnalyticsRow
                    label="Inference width"
                    value={
                      `${inferenceWidth} px`
                    }
                  />


                  <AnalyticsRow
                    label="Confidence threshold"
                    value={
                      confidence.toFixed(
                        2
                      )
                    }
                  />


                  <AnalyticsRow
                    label="Detection socket"
                    value={
                      socketConnected
                        ? "Streaming"
                        : "Idle"
                    }
                  />

                </div>

              </div>


              <div className="panel">

                <div className="panel-heading">

                  <div>

                    <h2>
                      Detection Activity
                    </h2>

                    <p>
                      Session totals
                    </p>

                  </div>

                </div>


                <div className="activity-bars">

                  <ActivityBar
                    label="Person observations"
                    value={
                      session.personObservations
                    }
                    max={
                      Math.max(
                        session.personObservations,
                        session.threatObservations,
                        1
                      )
                    }
                    type="blue"
                  />


                  <ActivityBar
                    label="Threat observations"
                    value={
                      session.threatObservations
                    }
                    max={
                      Math.max(
                        session.personObservations,
                        session.threatObservations,
                        1
                      )
                    }
                    type="red"
                  />


                  <ActivityBar
                    label="Tracked records"
                    value={
                      events.length
                    }
                    max={
                      Math.max(
                        events.length,
                        1
                      )
                    }
                    type="green"
                  />

                </div>

              </div>

            </div>

          </section>

        </div>


        {/* ==================================================
            SETTINGS
            ================================================== */}

        <div
          className={
            activePage ===
            "settings"
              ? "page-visible"
              : "page-hidden"
          }
        >

          <section className="page-section">

            <div className="settings-header">

              <span className="section-kicker">
                CONFIGURATION
              </span>

              <h2>
                Detection Settings
              </h2>

              <p>
                Controls for the active local
                detection pipeline.
              </p>

            </div>


            <div className="settings-layout">

              <div className="panel">

                <div className="settings-card-title">
                  Detection
                </div>


                <SettingRow
                  title="Confidence threshold"
                  description="Minimum confidence required for a returned detection."
                >

                  <div className="setting-slider">

                    <strong>
                      {confidence.toFixed(
                        2
                      )}
                    </strong>


                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={
                        confidence
                      }
                      onChange={(
                        event
                      ) =>
                        setConfidence(
                          Number(
                            event.target.value
                          )
                        )
                      }
                    />

                  </div>

                </SettingRow>


                <SettingRow
                  title="Track IDs"
                  description="Show ByteTrack IDs in the monitor and event history."
                >

                  <Toggle
                    checked={
                      showTrackIds
                    }
                    onChange={
                      setShowTrackIds
                    }
                  />

                </SettingRow>


                <SettingRow
                  title="Inference width"
                  description="Lower values reduce frame size and processing load."
                >

                  <select
                    className="setting-select"
                    value={
                      inferenceWidth
                    }
                    onChange={(
                      event
                    ) =>
                      setInferenceWidth(
                        Number(
                          event.target.value
                        )
                      )
                    }
                  >

                    <option value="480">
                      480 px
                    </option>

                    <option value="640">
                      640 px
                    </option>

                    <option value="768">
                      768 px
                    </option>

                    <option value="832">
                      832 px
                    </option>

                  </select>

                </SettingRow>

              </div>


              <div className="panel">

                <div className="settings-card-title">
                  Event Handling
                </div>


                <SettingRow
                  title="Session event persistence"
                  description="Current browser-session persistence setting."
                >

                  <Toggle
                    checked={
                      saveEvents
                    }
                    onChange={
                      setSaveEvents
                    }
                  />

                </SettingRow>


                <div className="settings-note">

                  <Activity
                    size={16}
                  />

                  <div>

                    <strong>
                      Local processing
                    </strong>

                    <span>
                      Frames are sent to the
                      local FastAPI service and
                      processed by YOLO.
                    </span>

                  </div>

                </div>

              </div>

            </div>

          </section>

        </div>


        {/* ==================================================
            ABOUT
            ================================================== */}

        <div
          className={
            activePage ===
            "about"
              ? "page-visible"
              : "page-hidden"
          }
        >

          <section className="page-section">

            <div className="about-hero panel">

              <div className="about-icon">

                <ShieldCheck
                  size={30}
                />

              </div>


              <div>

                <span className="section-kicker">
                  SECURITY PLATFORM
                </span>

                <h2>
                  Threat Detection YOLO11
                </h2>

                <p>
                  Real-time object detection,
                  tracking and monitoring.
                </p>

              </div>

            </div>


            <div className="about-grid">

              <InfoCard
                icon={
                  <Cpu
                    size={18}
                  />
                }
                label="MODEL"
                value={
                  modelInfo.name
                }
              />


              <InfoCard
                icon={
                  <Monitor
                    size={18}
                  />
                }
                label="DEVICE"
                value={
                  modelInfo.deviceName
                }
              />


              <InfoCard
                icon={
                  <Radio
                    size={18}
                  />
                }
                label="BACKEND"
                value={
                  backendConnected
                    ? "Connected"
                    : "Offline"
                }
              />


              <InfoCard
                icon={
                  <Users
                    size={18}
                  />
                }
                label="TRACKING"
                value="ByteTrack"
              />

            </div>


            <div className="panel about-details">

              <div>

                <span className="section-kicker">
                  MODEL
                </span>

                <h3>
                  Current development state
                </h3>

                <p>
                  The local environment currently
                  uses the pretrained YOLO11 model.
                  The custom two-class model will
                  automatically replace it when
                  <code>
                    models/best.pt
                  </code>
                  is available.
                </p>

              </div>


              <div className="class-list">

                <div>

                  <span className="class-dot blue" />

                  <strong>
                    normal_person
                  </strong>

                </div>


                <div>

                  <span className="class-dot red" />

                  <strong>
                    potentially_dangerous_person
                  </strong>

                </div>

              </div>

            </div>

          </section>

        </div>

      </main>

    </div>
  );
}


// ============================================================
// COMPONENTS
// ============================================================

function NavigationItem({
  icon,
  label,
  active,
  badge,
  onClick,
}) {

  return (
    <button
      type="button"
      className={
        active
          ? "navigation-item active"
          : "navigation-item"
      }
      onClick={onClick}
    >

      {icon}

      <span>
        {label}
      </span>


      {badge !==
        null &&
        badge !==
          undefined && (

          <span className="nav-badge">
            {badge}
          </span>

        )}

    </button>
  );
}


function StatusRow({
  label,
  value,
  state,
}) {

  return (
    <div className="status-row">

      <span>
        {label}
      </span>


      <strong
        className={state}
      >

        <span className="status-dot" />

        {value}

      </strong>

    </div>
  );
}


function MetricCard({
  icon,
  title,
  value,
  detail,
  type,
}) {

  return (
    <div
      className={
        `metric-card ${type}`
      }
    >

      <div className="metric-card-icon">
        {icon}
      </div>


      <span className="metric-card-title">
        {title}
      </span>


      <strong className="metric-card-value">
        {value}
      </strong>


      <span className="metric-card-detail">
        {detail}
      </span>

    </div>
  );
}


function EventRow({
  event,
  showTrackIds,
}) {

  return (
    <div className="event-row">

      <div className="event-time">
        {event.time}
      </div>


      <div
        className={
          event.isThreat
            ? "event-symbol threat"
            : "event-symbol"
        }
      >

        {event.isThreat ? (
          <AlertTriangle
            size={14}
          />
        ) : (
          <Users
            size={14}
          />
        )}

      </div>


      <div className="event-info">

        <strong
          className={
            event.isThreat
              ? "threat-text"
              : ""
          }
        >

          {event.type}


          {showTrackIds &&
            event.trackId !==
              null && (

              <small className="track-badge">

                P-
                {String(
                  event.trackId
                ).padStart(
                  2,
                  "0"
                )}

              </small>
            )}

        </strong>


        <span>
          {event.className}
        </span>

      </div>


      <strong className="event-confidence">

        {(
          event.confidence *
          100
        ).toFixed(
          1
        )}
        %

      </strong>

    </div>
  );
}


function AnalyticsRow({
  label,
  value,
}) {

  return (
    <div className="analytics-row">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

    </div>
  );
}


function ActivityBar({
  label,
  value,
  max,
  type,
}) {

  const percentage =
    max > 0
      ? Math.min(
          100,
          (value /
            max) *
            100
        )
      : 0;


  return (
    <div className="activity-bar">

      <div className="activity-bar-head">

        <span>
          {label}
        </span>

        <strong>
          {value}
        </strong>

      </div>


      <div className="activity-track">

        <div
          className={
            `activity-fill ${type}`
          }
          style={{
            width:
              `${percentage}%`,
          }}
        />

      </div>

    </div>
  );
}


function SettingRow({
  title,
  description,
  children,
}) {

  return (
    <div className="setting-row">

      <div>

        <strong>
          {title}
        </strong>

        <span>
          {description}
        </span>

      </div>


      <div className="setting-value">
        {children}
      </div>

    </div>
  );
}


function Toggle({
  checked,
  onChange,
}) {

  return (
    <button
      type="button"
      className={
        checked
          ? "toggle checked"
          : "toggle"
      }
      onClick={() =>
        onChange(
          !checked
        )
      }
      aria-label="Toggle setting"
    >

      <span />

    </button>
  );
}


function InfoCard({
  icon,
  label,
  value,
}) {

  return (
    <div className="info-card">

      <div className="info-card-icon">
        {icon}
      </div>


      <div>

        <span>
          {label}
        </span>

        <strong>
          {value}
        </strong>

      </div>

    </div>
  );
}


export default App;