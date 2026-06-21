const TOAST_DURATION_MS = 6000;

export function applyPrimereButtonStyle(widget, options = {}) {
    const {
        height = 32,
        radius = 6,
        margin = 15,
        bgColor = "#771a1a",
        activeColor = "#932424",
        textColor = "#dad570",
        fontSize = "bold 15px sans-serif",
        label = null,
    } = options;

    widget.computeSize = () => [0, height];
    widget.draw = function (ctx, node, widget_width, y) {
        ctx.save();
        ctx.fillStyle = this.clicked ? activeColor : bgColor;
        if (this.clicked) {
            this.clicked = false;
            node.setDirtyCanvas?.(true);
        }
        ctx.beginPath();
        ctx.roundRect(margin, y, widget_width - margin * 2, height, radius);
        ctx.fill();
        ctx.fillStyle = textColor;
        ctx.font = fontSize;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label ?? this.name, widget_width * 0.5, y + height * 0.5);
        ctx.restore();
    };
}

export function applyPrimereTwinButtonStyle(widget, options = {}) {
    const {
        height = 30,
        radius = 6,
        margin = 15,
        gap = 4,
        bgColor = "#274e7a",
        activeColor = "#366294",
        textColor = "#ffffff",
        fontSize = "bold 13px sans-serif",
        leftLabel = "Left",
        rightLabel = "Right",
    } = options;

    widget.computeSize = () => [0, height + 4];

    widget.draw = function (ctx, node, widget_width, y) {
        ctx.save();
        const halfW = (widget_width - margin * 2 - gap) / 2;
        const leftX = margin;
        const rightX = leftX + halfW + gap;

        ctx.fillStyle = this.__twinClickedSide === "left" ? activeColor : bgColor;
        ctx.beginPath();
        ctx.roundRect(leftX, y + 2, halfW, height, radius);
        ctx.fill();
        ctx.fillStyle = textColor;
        ctx.font = fontSize;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(leftLabel, leftX + halfW / 2, y + 2 + height / 2);

        ctx.fillStyle = this.__twinClickedSide === "right" ? activeColor : bgColor;
        ctx.beginPath();
        ctx.roundRect(rightX, y + 2, halfW, height, radius);
        ctx.fill();
        ctx.fillStyle = textColor;
        ctx.fillText(rightLabel, rightX + halfW / 2, y + 2 + height / 2);

        ctx.restore();

        if (this.__twinClickedSide) {
            this.__twinClickedSide = null;
            node.setDirtyCanvas?.(true);
        }
    };

    widget.__twinButtonCfg = { margin, gap };
}

export function getTwinButtonSide(widget, node, posX) {
    const cfg = widget.__twinButtonCfg || { margin: 15, gap: 4 };
    const halfW = (node.size[0] - cfg.margin * 2 - cfg.gap) / 2;
    const relX = posX - cfg.margin;
    return relX < halfW + cfg.gap / 2 ? "left" : "right";
}

export function showToast(status, message) {
    let container = document.getElementById("primere-toast-container");
    if (!container) {
        container = document.createElement("div");
        container.id = "primere-toast-container";
        Object.assign(container.style, {
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: "99999",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            pointerEvents: "none",
        });
        document.body.appendChild(container);
    }

    const isSuccess = status === "success";
    const accentColor = isSuccess ? "#4caf50" : "#f44336";
    const icon = isSuccess ? "✔" : "✖";

    const toast = document.createElement("div");
    Object.assign(toast.style, {
        position: "relative",
        overflow: "hidden",
        background: "#1e1e1e",
        border: `1px solid ${accentColor}`,
        borderLeft: `4px solid ${accentColor}`,
        borderRadius: "6px",
        padding: "10px 14px 10px 12px",
        minWidth: "280px",
        maxWidth: "440px",
        display: "flex",
        alignItems: "flex-start",
        gap: "10px",
        pointerEvents: "all",
        boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
        opacity: "0",
        transform: "translateY(-12px)",
        transition: "opacity 0.25s ease, transform 0.25s ease",
        fontFamily: "sans-serif",
        fontSize: "15px",
        color: "#e0e0e0",
        cursor: "default",
    });

    const iconEl = document.createElement("span");
    iconEl.textContent = icon;
    Object.assign(iconEl.style, { color: accentColor, fontSize: "15px", flexShrink: "0", lineHeight: "1.5" });

    const textEl = document.createElement("span");
    textEl.textContent = message;
    Object.assign(textEl.style, { flex: "1", lineHeight: "1.5", wordBreak: "normal", overflowWrap: "break-word" });

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    Object.assign(closeBtn.style, {
        background: "none", border: "none", color: "#888", cursor: "pointer",
        fontSize: "13px", padding: "0 0 0 6px", flexShrink: "0", lineHeight: "1.5",
    });
    closeBtn.addEventListener("mouseover", () => { closeBtn.style.color = "#ccc"; });
    closeBtn.addEventListener("mouseout",  () => { closeBtn.style.color = "#888"; });

    const progress = document.createElement("div");
    Object.assign(progress.style, {
        position: "absolute", bottom: "0", left: "0",
        height: "3px", width: "100%",
        background: accentColor,
        transformOrigin: "left",
        transform: "scaleX(1)",
        transition: `transform ${TOAST_DURATION_MS / 1000}s linear`,
    });

    toast.appendChild(iconEl);
    toast.appendChild(textEl);
    toast.appendChild(closeBtn);
    toast.appendChild(progress);
    container.appendChild(toast);

    requestAnimationFrame(() => requestAnimationFrame(() => {
        toast.style.opacity = "1";
        toast.style.transform = "translateY(0)";
        progress.style.transform = "scaleX(0)";
    }));

    function dismiss() {
        clearTimeout(timer);
        toast.style.opacity = "0";
        toast.style.transform = "translateY(-12px)";
        setTimeout(() => toast.remove(), 280);
    }

    const timer = setTimeout(dismiss, TOAST_DURATION_MS);
    closeBtn.addEventListener("click", dismiss);
}

export function applyPrimereDualButtonStyle(widget, options = {}) {
    const {
        height = 32,
        radius = 6,
        margin = 15,
        gap = 4,
        leftLabel = "Left",
        rightLabel = "Right",
        leftBg = "#2d6e2d",
        leftActive = "#3d8e3d",
        rightBg = "#771a1a",
        rightActive = "#932424",
        textColor = "#ffffff",
        fontSize = "bold 13px sans-serif",
    } = options;

    widget.computeSize = () => [0, height + 4];

    widget.draw = function (ctx, node, widget_width, y) {
        ctx.save();
        const halfW = (widget_width - margin * 2 - gap) / 2;
        const leftX = margin;
        const rightX = leftX + halfW + gap;

        ctx.fillStyle = this.__dualClickedSide === "left" ? leftActive : leftBg;
        ctx.beginPath();
        ctx.roundRect(leftX, y + 2, halfW, height, radius);
        ctx.fill();
        ctx.fillStyle = textColor;
        ctx.font = fontSize;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(leftLabel, leftX + halfW / 2, y + 2 + height / 2);

        ctx.fillStyle = this.__dualClickedSide === "right" ? rightActive : rightBg;
        ctx.beginPath();
        ctx.roundRect(rightX, y + 2, halfW, height, radius);
        ctx.fill();
        ctx.fillStyle = textColor;
        ctx.fillText(rightLabel, rightX + halfW / 2, y + 2 + height / 2);

        ctx.restore();

        if (this.__dualClickedSide) {
            this.__dualClickedSide = null;
            node.setDirtyCanvas?.(true);
        }
    };

    widget.__dualButtonCfg = { margin, gap };
}

export function getDualButtonSide(widget, node, posX) {
    const cfg = widget.__dualButtonCfg || { margin: 15, gap: 4 };
    const halfW = (node.size[0] - cfg.margin * 2 - cfg.gap) / 2;
    const relX = posX - cfg.margin;
    return relX < halfW + cfg.gap / 2 ? "left" : "right";
}

export function applyPrimereTripleButtonStyle(widget, options = {}) {
    const {
        height = 30,
        radius = 6,
        margin = 15,
        gap = 4,
        bgColor = "#274e7a",
        activeColor = "#366294",
        textColor = "#ffffff",
        fontSize = "bold 13px sans-serif",
        leftLabel = "Left",
        centerLabel = "Center",
        rightLabel = "Right",
    } = options;

    widget.computeSize = () => [0, height + 4];

    widget.draw = function (ctx, node, widget_width, y) {
        ctx.save();
        const thirds = (widget_width - margin * 2 - gap * 2) / 3;
        const leftX = margin;
        const centerX = leftX + thirds + gap;
        const rightX = centerX + thirds + gap;

        ctx.fillStyle = this.__tripleClickedSide === "left" ? activeColor : bgColor;
        ctx.beginPath();
        ctx.roundRect(leftX, y + 2, thirds, height, radius);
        ctx.fill();
        ctx.fillStyle = textColor;
        ctx.font = fontSize;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(leftLabel, leftX + thirds / 2, y + 2 + height / 2);

        ctx.fillStyle = this.__tripleClickedSide === "center" ? activeColor : bgColor;
        ctx.beginPath();
        ctx.roundRect(centerX, y + 2, thirds, height, radius);
        ctx.fill();
        ctx.fillStyle = textColor;
        ctx.fillText(centerLabel, centerX + thirds / 2, y + 2 + height / 2);

        ctx.fillStyle = this.__tripleClickedSide === "right" ? activeColor : bgColor;
        ctx.beginPath();
        ctx.roundRect(rightX, y + 2, thirds, height, radius);
        ctx.fill();
        ctx.fillStyle = textColor;
        ctx.fillText(rightLabel, rightX + thirds / 2, y + 2 + height / 2);

        ctx.restore();

        if (this.__tripleClickedSide) {
            this.__tripleClickedSide = null;
            node.setDirtyCanvas?.(true);
        }
    };

    widget.__tripleButtonCfg = { margin, gap };
}

export function getTripleButtonSide(widget, node, posX) {
    const cfg = widget.__tripleButtonCfg || { margin: 15, gap: 4 };
    const thirds = (node.size[0] - cfg.margin * 2 - cfg.gap * 2) / 3;
    const relX = posX - cfg.margin;
    if (relX < thirds + cfg.gap / 2) return "left";
    if (relX < thirds * 2 + cfg.gap + cfg.gap / 2) return "center";
    return "right";
}
