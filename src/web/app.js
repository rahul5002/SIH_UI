document.addEventListener("DOMContentLoaded", () => {
    // State variables
    let currentDocFile = null;
    let currentSelfieFile = null;

    // View Navigation
    window.navigateTo = function(viewId) {
        document.querySelectorAll(".view-panel").forEach(p => p.classList.remove("active"));
        document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));

        const targetPanel = document.getElementById(viewId);
        if (targetPanel) targetPanel.classList.add("active");

        const targetNav = document.querySelector(`.nav-item[data-view="${viewId}"]`);
        if (targetNav) targetNav.classList.add("active");

        // Update header greeting based on view
        const pageGreeting = document.getElementById("pageGreeting");
        const pageSubtext = document.getElementById("pageSubtext");

        if (viewId === "scanDetectView") {
            pageGreeting.textContent = "AI Document Screening & Biometric Engine";
            pageSubtext.textContent = "Multi-stage digital forensics, ELA compression heatmaps, and facial verification.";
        } else if (viewId === "recordsView") {
            pageGreeting.textContent = "National ID Records Registry";
            pageSubtext.textContent = "Search, filter, and inspect verified citizen identity documents.";
        } else if (viewId === "documentsView") {
            pageGreeting.textContent = "Document Repository";
            pageSubtext.textContent = "Cryptographically hashed document storage & audit records.";
        } else if (viewId === "verificationsView") {
            pageGreeting.textContent = "Verification Audit Logs";
            pageSubtext.textContent = "Live log of all AI screening evaluations and checkpoint decisions.";
        } else {
            pageGreeting.textContent = "Welcome, Admin User";
            pageSubtext.textContent = "Monitor verification activities and manage records securely.";
        }
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // Nav Item Click Handlers
    document.querySelectorAll(".nav-item[data-view]").forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const viewId = item.getAttribute("data-view");
            navigateTo(viewId);
        });
    });

    // Helper to load sample from recent table directly into AI scanner
    window.loadDocumentForScreening = function(docName, docType) {
        showToast(`Loading ${docName} for AI Screening...`);
        navigateTo("scanDetectView");

        // Map docType to scenario
        let scenario = "CLEAN";
        if (docName.includes("Neha")) scenario = "TEXT_DATE";
        else if (docName.includes("Amit")) scenario = "FAKE_STAMP";
        else if (docName.includes("Suresh")) scenario = "PHOTO_REPLACEMENT";

        triggerSyntheticScenario(scenario);
    };

    window.inspectRecord = function(docType, name, idNum) {
        showToast(`Inspecting record: ${name} (${docType})`);
        navigateTo("scanDetectView");
        triggerSyntheticScenario("CLEAN");
    };

    // DOM Elements for Screener
    const docFileInput = document.getElementById("docFileInput");
    const selfieFileInput = document.getElementById("selfieFileInput");
    const docDropzone = document.getElementById("docDropzone");
    const docPreviewContainer = document.getElementById("docPreviewContainer");
    const docPreviewImg = document.getElementById("docPreviewImg");
    const btnClearDoc = document.getElementById("btnClearDoc");
    const selfiePreviewImg = document.getElementById("selfiePreviewImg");
    const selfiePlaceholderIcon = document.getElementById("selfiePlaceholderIcon");
    const btnRunScreening = document.getElementById("btnRunScreening");

    const visualizerStage = document.getElementById("visualizerStage");
    const stagePlaceholder = document.getElementById("stagePlaceholder");
    const imageOverlayWrapper = document.getElementById("imageOverlayWrapper");
    const stageDocImg = document.getElementById("stageDocImg");
    const stageHeatmapImg = document.getElementById("stageHeatmapImg");
    const evidenceBoxesLayer = document.getElementById("evidenceBoxesLayer");
    const toggleHeatmap = document.getElementById("toggleHeatmap");

    const meterEla = document.getElementById("meterEla");
    const valEla = document.getElementById("valEla");
    const meterNoise = document.getElementById("meterNoise");
    const valNoise = document.getElementById("valNoise");
    const meterSplicing = document.getElementById("meterSplicing");
    const valSplicing = document.getElementById("valSplicing");
    const meterFont = document.getElementById("meterFont");
    const valFont = document.getElementById("valFont");
    const reasonsList = document.getElementById("reasonsList");

    const verdictBanner = document.getElementById("verdictBanner");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictSubtitle = document.getElementById("verdictSubtitle");
    const verdictIcon = document.getElementById("verdictIcon");
    const riskPill = document.getElementById("riskPill");

    const ocrTableBody = document.getElementById("ocrTableBody");
    const mrzDisplay = document.getElementById("mrzDisplay");
    const checkList = document.getElementById("checkList");
    const violationsList = document.getElementById("violationsList");

    const bioDocFace = document.getElementById("bioDocFace");
    const bioLiveFace = document.getElementById("bioLiveFace");
    const bioSimScore = document.getElementById("bioSimScore");
    const bioLivenessScore = document.getElementById("bioLivenessScore");
    const bioVerdictTag = document.getElementById("bioVerdictTag");

    // Webcam Elements
    const btnOpenWebcam = document.getElementById("btnOpenWebcam");
    const webcamModal = document.getElementById("webcamModal");
    const btnCloseWebcam = document.getElementById("btnCloseWebcam");
    const webcamVideo = document.getElementById("webcamVideo");
    const webcamCanvas = document.getElementById("webcamCanvas");
    const btnCaptureSelfie = document.getElementById("btnCaptureSelfie");
    let mediaStream = null;

    const benchmarkModal = document.getElementById("benchmarkModal");
    const btnOpenBenchmark = document.getElementById("btnOpenBenchmark");
    const btnCloseBenchmark = document.getElementById("btnCloseBenchmark");
    const btnStartBenchmark = document.getElementById("btnStartBenchmark");

    const toast = document.getElementById("toast");

    // Toast helper
    function showToast(msg) {
        toast.textContent = msg;
        toast.classList.add("show");
        setTimeout(() => toast.classList.remove("show"), 3500);
    }

    // Tabs switching
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            const target = btn.getAttribute("data-tab");
            const targetEl = document.getElementById(target);
            if (targetEl) targetEl.classList.add("active");
        });
    });

    // File Upload Handlers
    docFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleDocSelection(file);
    });

    selfieFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleSelfieSelection(file);
    });

    // Drag and Drop
    docDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        docDropzone.classList.add("dragover");
    });

    docDropzone.addEventListener("dragleave", () => {
        docDropzone.classList.remove("dragover");
    });

    docDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        docDropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleDocSelection(e.dataTransfer.files[0]);
        }
    });

    btnClearDoc.addEventListener("click", (e) => {
        e.stopPropagation();
        currentDocFile = null;
        docFileInput.value = "";
        docPreviewContainer.classList.add("hidden");
        document.querySelector(".dropzone-content").classList.remove("hidden");
    });

    function handleDocSelection(file) {
        currentDocFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            docPreviewImg.src = e.target.result;
            docPreviewContainer.classList.remove("hidden");
            document.querySelector(".dropzone-content").classList.add("hidden");
        };
        reader.readAsDataURL(file);
        showToast("Document loaded: " + file.name);
    }

    function handleSelfieSelection(file) {
        currentSelfieFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            selfiePreviewImg.src = e.target.result;
            selfiePreviewImg.classList.remove("hidden");
            selfiePlaceholderIcon.classList.add("hidden");
        };
        reader.readAsDataURL(file);
        showToast("Passenger selfie attached");
    }

    // Real-Time Webcam Streaming & Capture
    btnOpenWebcam.addEventListener("click", async () => {
        try {
            webcamModal.classList.remove("hidden");
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });
            webcamVideo.srcObject = mediaStream;
            showToast("Camera online. Align face inside reticle.");
        } catch (err) {
            console.error("Webcam access error:", err);
            showToast("⚠️ Could not access camera. Please allow camera permissions or upload an image.");
            closeWebcamModal();
        }
    });

    function closeWebcamModal() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        webcamVideo.srcObject = null;
        webcamModal.classList.add("hidden");
    }

    btnCloseWebcam.addEventListener("click", closeWebcamModal);

    btnCaptureSelfie.addEventListener("click", () => {
        if (!mediaStream) return;

        const width = webcamVideo.videoWidth || 640;
        const height = webcamVideo.videoHeight || 480;

        webcamCanvas.width = width;
        webcamCanvas.height = height;
        const ctx = webcamCanvas.getContext("2d");

        // Flip horizontally to match the mirrored selfie view
        ctx.translate(width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(webcamVideo, 0, 0, width, height);

        webcamCanvas.toBlob((blob) => {
            const capturedFile = new File([blob], "live_passenger_selfie.png", { type: "image/png" });
            handleSelfieSelection(capturedFile);
            closeWebcamModal();
            showToast("📸 Real-time selfie captured successfully!");
        }, "image/png");
    });

    // Quick Synthetic Scenario Generator Trigger
    async function triggerSyntheticScenario(scenario) {
        showToast(`Generating ${scenario} synthetic sample...`);

        try {
            const formData = new FormData();
            formData.append("tampering_scenario", scenario);

            const res = await fetch("/api/v1/generate-synthetic-data", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            // Convert base64 to Blob
            const fetchRes = await fetch(data.document_base64);
            const blob = await fetchRes.blob();
            const file = new File([blob], `synthetic_${scenario.toLowerCase()}.png`, { type: "image/png" });

            handleDocSelection(file);

            // If avatar provided, attach as selfie
            if (data.avatar_base64) {
                const avRes = await fetch(data.avatar_base64);
                const avBlob = await avRes.blob();
                const avFile = new File([avBlob], "avatar.png", { type: "image/png" });
                handleSelfieSelection(avFile);
            }

            // Automatically trigger screening
            setTimeout(() => runScreening(), 400);

        } catch (err) {
            console.error(err);
            showToast("Failed to generate synthetic sample.");
        }
    }

    document.querySelectorAll(".btn-scenario").forEach(btn => {
        btn.addEventListener("click", () => {
            const scenario = btn.getAttribute("data-scenario");
            triggerSyntheticScenario(scenario);
        });
    });

    // Heatmap visibility toggle
    toggleHeatmap.addEventListener("change", () => {
        stageHeatmapImg.style.opacity = toggleHeatmap.checked ? "0.85" : "0.0";
    });

    // Run Full Screening
    btnRunScreening.addEventListener("click", runScreening);

    async function runScreening() {
        if (!currentDocFile) {
            showToast("⚠️ Please upload or select a document first!");
            return;
        }

        btnRunScreening.disabled = true;
        btnRunScreening.innerHTML = `<span class="btn-icon">⚙️</span> Processing AI Pipeline...`;
        showToast("Screening document through 5-stage AI pipeline...");

        const formData = new FormData();
        formData.append("document", currentDocFile);
        if (currentSelfieFile) {
            formData.append("live_selfie", currentSelfieFile);
        }

        try {
            const res = await fetch("/api/v1/screen-document", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error("Screening request failed");
            const data = await res.json();

            renderScreeningResults(data);
            showToast(`Screening Completed in ${data.processing_time_ms} ms!`);

        } catch (err) {
            console.error(err);
            showToast("❌ Error running screening pipeline");
        } finally {
            btnRunScreening.disabled = false;
            btnRunScreening.innerHTML = `<span class="btn-icon">🔍</span> Run Full AI Forensic Screening`;
        }
    }

    function renderScreeningResults(data) {
        // 1. Render Visualizer & Heatmap
        stagePlaceholder.classList.add("hidden");
        imageOverlayWrapper.classList.remove("hidden");

        const reader = new FileReader();
        reader.onload = (e) => {
            stageDocImg.src = e.target.result;
        };
        reader.readAsDataURL(currentDocFile);

        if (data.tampering_report.heatmap_base64) {
            stageHeatmapImg.src = data.tampering_report.heatmap_base64;
        }

        // Draw Evidence Bounding Boxes
        evidenceBoxesLayer.innerHTML = "";
        const evidenceRegions = data.tampering_report.evidence_regions || [];
        
        stageDocImg.onload = () => {
            const dispW = stageDocImg.clientWidth;
            const dispH = stageDocImg.clientHeight;
            const natW = stageDocImg.naturalWidth || 1000;
            const natH = stageDocImg.naturalHeight || 700;

            const scaleX = dispW / natW;
            const scaleY = dispH / natH;

            evidenceRegions.forEach(ev => {
                const box = document.createElement("div");
                box.className = "evidence-box";
                box.style.left = `${ev.bbox.x * scaleX}px`;
                box.style.top = `${ev.bbox.y * scaleY}px`;
                box.style.width = `${ev.bbox.width * scaleX}px`;
                box.style.height = `${ev.bbox.height * scaleY}px`;
                box.title = `Evidence: ${ev.reason}`;
                box.addEventListener("click", () => {
                    alert(`Forensic Finding:\n${ev.reason}\nConfidence: ${Math.round((ev.confidence || 0.85)*100)}%`);
                });
                evidenceBoxesLayer.appendChild(box);
            });
        };

        // 2. Meters
        const m = data.tampering_report.module_scores || {};
        meterEla.style.width = `${m.ela_score || 0}%`;
        valEla.textContent = `${m.ela_score || 0}%`;

        meterNoise.style.width = `${m.noise_inconsistency || 0}%`;
        valNoise.textContent = `${m.noise_inconsistency || 0}%`;

        meterSplicing.style.width = `${m.splicing_boundary_score || 0}%`;
        valSplicing.textContent = `${m.splicing_boundary_score || 0}%`;

        meterFont.style.width = `${m.font_anomaly_score || 0}%`;
        valFont.textContent = `${m.font_anomaly_score || 0}%`;

        // Reasons
        reasonsList.innerHTML = "";
        (data.tampering_report.reasons || []).forEach(r => {
            const li = document.createElement("li");
            li.textContent = r;
            reasonsList.appendChild(li);
        });

        // 3. Verdict Banner
        verdictBanner.className = "verdict-banner";
        if (data.overall_verdict.includes("CLEAN") || data.overall_verdict.includes("PASSED")) {
            verdictBanner.classList.add("clean");
            verdictIcon.textContent = "✅";
            verdictTitle.textContent = "DOCUMENT CLEARED (GENUINE)";
            verdictSubtitle.textContent = "All security checksums & forensic checks passed.";
        } else if (data.overall_verdict.includes("EXPIRED")) {
            verdictBanner.classList.add("expired");
            verdictIcon.textContent = "⚠️";
            verdictTitle.textContent = "REJECTED: EXPIRED DOCUMENT";
            verdictSubtitle.textContent = "Travel document is past its validity expiry date.";
        } else {
            verdictBanner.classList.add("tampered");
            verdictIcon.textContent = "🚨";
            verdictTitle.textContent = "FLAGGED: TAMPERING DETECTED";
            verdictSubtitle.textContent = "Digital forgery, altered text, or spliced photo detected.";
        }
        riskPill.textContent = `RISK: ${data.overall_risk_score}%`;

        // 4. OCR Fields Table
        ocrTableBody.innerHTML = "";
        const f = data.extracted_fields || {};
        const confs = data.field_confidences || {};
        const fieldKeys = [
            ["Document Type", f.document_type],
            ["Document Number", f.document_number],
            ["Full Name", f.full_name],
            ["Nationality", f.nationality],
            ["Date of Birth", f.date_of_birth],
            ["Gender", f.gender],
            ["Date of Expiry", f.date_of_expiry]
        ];

        fieldKeys.forEach(([k, v]) => {
            const tr = document.createElement("tr");
            const conf = confs[k.toLowerCase().replace(/ /g, "_")] || data.ocr_confidence || 0.88;
            tr.innerHTML = `
                <td><strong>${k}</strong></td>
                <td>${v || '<span class="text-muted">--</span>'}</td>
                <td><span style="color: var(--brand-blue); font-weight: bold;">${Math.round(conf * 100)}%</span></td>
            `;
            ocrTableBody.appendChild(tr);
        });

        // MRZ String
        if (data.mrz_details && data.mrz_details.raw_mrz) {
            mrzDisplay.textContent = data.mrz_details.raw_mrz;
        } else {
            mrzDisplay.textContent = "No MRZ decoded";
        }

        // 5. Checksum Checklist
        const cs = (data.validation_report && data.validation_report.checksum_validation && data.validation_report.checksum_validation.details) || {};
        checkList.innerHTML = `
            <div class="check-item"><span class="check-icon">${cs.document_number_valid ? "🟢" : "🔴"}</span> Document Number Checksum</div>
            <div class="check-item"><span class="check-icon">${cs.date_of_birth_valid ? "🟢" : "🔴"}</span> Date of Birth Checksum</div>
            <div class="check-item"><span class="check-icon">${cs.date_of_expiry_valid ? "🟢" : "🔴"}</span> Expiry Date Checksum</div>
            <div class="check-item"><span class="check-icon">${cs.composite_valid ? "🟢" : "🔴"}</span> Composite Final Checksum</div>
        `;

        // Violations
        violationsList.innerHTML = "";
        const viols = data.validation_report.violations || [];
        if (viols.length === 0) {
            violationsList.innerHTML = "<li>None detected (100% Valid Standard)</li>";
        } else {
            viols.forEach(v => {
                const li = document.createElement("li");
                li.style.color = "var(--status-danger)";
                li.textContent = v;
                violationsList.appendChild(li);
            });
        }

        // 6. Biometrics
        if (data.face_report && data.face_report.success) {
            bioDocFace.src = data.face_report.doc_face_base64 || "";
            bioLiveFace.src = data.face_report.live_face_base64 || "";
            bioDocFace.classList.remove("placeholder-face");
            bioLiveFace.classList.remove("placeholder-face");

            const sim = Math.round(data.face_report.similarity_score * 100);
            bioSimScore.textContent = `${sim}%`;
            bioLivenessScore.textContent = data.face_report.liveness_score ? `${Math.round(data.face_report.liveness_score * 100)}% (PASSED)` : "N/A";

            bioVerdictTag.textContent = data.face_report.is_match ? "✅ BIOMETRIC IDENTITY MATCH" : "❌ BIOMETRIC MISMATCH (IMPOSTOR)";
            bioVerdictTag.style.background = data.face_report.is_match ? "var(--stat-green-bg)" : "var(--status-danger-bg)";
            bioVerdictTag.style.color = data.face_report.is_match ? "#15803d" : "#dc2626";
        } else {
            bioSimScore.textContent = "--";
            bioLivenessScore.textContent = "--";
            bioVerdictTag.textContent = "No Live Selfie Provided";
            bioVerdictTag.style.background = "#f1f5f9";
            bioVerdictTag.style.color = "var(--text-muted)";
        }
    }

    // Benchmark Modal Events
    btnOpenBenchmark.addEventListener("click", () => {
        benchmarkModal.classList.remove("hidden");
    });

    btnCloseBenchmark.addEventListener("click", () => {
        benchmarkModal.classList.add("hidden");
    });

    btnStartBenchmark.addEventListener("click", async () => {
        btnStartBenchmark.disabled = true;
        btnStartBenchmark.textContent = "Running Benchmark Suite...";
        showToast("Generating batch and calculating AI accuracy...");

        try {
            const res = await fetch("/api/v1/run-benchmark?num_samples=8");
            const data = await res.json();

            document.getElementById("bAccuracy").textContent = `${Math.round(data.metrics.accuracy * 100)}%`;
            document.getElementById("bPrecision").textContent = `${Math.round(data.metrics.precision * 100)}%`;
            document.getElementById("bRecall").textContent = `${Math.round(data.metrics.recall * 100)}%`;
            document.getElementById("bF1").textContent = `${Math.round(data.metrics.f1_score * 100)}%`;

            const tbody = document.getElementById("benchmarkTableBody");
            tbody.innerHTML = "";

            data.detailed_samples.forEach(s => {
                const tr = document.createElement("tr");
                const isCorrect = s.ground_truth_tampered === s.predicted_tampered;
                tr.innerHTML = `
                    <td>#${s.sample_id}</td>
                    <td>${s.scenario}</td>
                    <td>${s.ground_truth_tampered ? "Tampered" : "Clean"}</td>
                    <td>${s.predicted_tampered ? "Tampered" : "Clean"}</td>
                    <td>${s.tampering_score}%</td>
                    <td><span style="color: ${isCorrect ? 'var(--stat-green)' : 'var(--status-danger)'}">${isCorrect ? '✅ MATCH' : '❌ MISMATCH'}</span></td>
                `;
                tbody.appendChild(tr);
            });

            showToast("Benchmark completed successfully!");

        } catch (err) {
            console.error(err);
            showToast("Benchmark execution failed.");
        } finally {
            btnStartBenchmark.disabled = false;
            btnStartBenchmark.textContent = "Run 8-Sample Evaluation Batch";
        }
    });
});
