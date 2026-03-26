const TOAST_DURATION_MS = 6000;
const BTN_HEIGHT = 32;
const BTN_COLOR = "#771a1a";
const BTN_COLOR_ACTIVE = "#932424";
const BTN_RADIUS = 6;
const BTN_FONT = "bold 15px sans-serif";

export function applyPrimereButtonStyle(widget) {
    widget.computeSize = () => [0, BTN_HEIGHT];
    widget.draw = function (ctx, node, widget_width, y) {
        ctx.save();
        const margin = 15;
        ctx.fillStyle = this.clicked ? BTN_COLOR_ACTIVE : BTN_COLOR;
        if (this.clicked) {
            this.clicked = false;
            node.setDirtyCanvas?.(true);
        }
        ctx.beginPath();
        ctx.roundRect(margin, y, widget_width - margin * 2, BTN_HEIGHT, BTN_RADIUS);
        ctx.fill();
        ctx.fillStyle = "#dad570";
        ctx.font = BTN_FONT;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(this.name, widget_width * 0.5, y + BTN_HEIGHT * 0.5);
        ctx.restore();
    };
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
