/**
 * Ember Resolution (MP): a live preview of the resolution the node will actually output.
 *
 *   1600 x 2784  ·  4.454 MP (+0.1%)  ·  0.5747
 *
 * The line is display only:
 * - widget.serialize = false keeps it out of the saved workflow's widgets_values, so a
 *   saved node still carries exactly its 3 real values;
 * - options.serialize = false keeps it out of the API prompt.
 *
 * The numbers come from lib/resolution_mp_preview.js, which is tested to match the
 * Python node exactly (tests/resolution_preview). When a value is only known at run
 * time (ratio taken from the image, or a widget driven by a link) it says so rather
 * than guessing.
 */

import { app } from "../../scripts/app.js";
import { previewText } from "./lib/resolution_mp_preview.js";

const NODE_CLASS = "EmberResolutionMP";
const WATCHED = ["megapixels", "aspect_ratio", "multiple_of"];
const PREVIEW_NAME = "preview";

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function linkedInput(node, name) {
    return node.inputs?.find((input) => input.name === name && input.link != null);
}

function describe(node) {
    const values = {};
    for (const name of WATCHED) {
        if (linkedInput(node, name)) return `${name} from input — computed at run time`;
        const w = widget(node, name);
        if (!w) return "widgets not found";
        values[name] = w.value;
    }
    try {
        return previewText(values.megapixels, values.aspect_ratio, values.multiple_of);
    } catch (error) {
        console.error("[Ember Resolution] preview failed", error);
        return "preview unavailable — see console";
    }
}

function refresh(node) {
    const preview = node._emberResolutionPreview;
    if (!preview) return;
    const text = describe(node);
    if (preview.value !== text) {
        preview.value = text;
        node.setDirtyCanvas?.(true, true);
    }
}

function attach(node) {
    if (node._emberResolutionPreview) return;
    if (!WATCHED.every((name) => widget(node, name))) return false;

    // Not disabled: some frontend versions draw a disabled text widget without its value.
    // Edits are simply overwritten with the computed line.
    const preview = node.addWidget("text", PREVIEW_NAME, "", () => refresh(node), { serialize: false });
    preview.serialize = false; // what LGraphNode.serialize checks (options.serialize alone is not enough)
    node._emberResolutionPreview = preview;

    for (const name of WATCHED) {
        const w = widget(node, name);
        const previous = w.callback;
        w.callback = function (...args) {
            const result = previous?.apply(this, args);
            refresh(node);
            return result;
        };
    }

    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = onConfigure?.apply(this, args);
        refresh(node); // a loaded workflow sets widget values without firing callbacks
        return result;
    };

    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = onConnectionsChange?.apply(this, args);
        refresh(node);
        return result;
    };

    refresh(node);
    return true;
}

app.registerExtension({
    name: "ember.ResolutionMPPreview",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        if (attach(node) === false) {
            // Widgets not in place yet on this frontend: try again on the next frames.
            let tries = 0;
            const retry = () => {
                if (attach(node) === false && ++tries < 20) requestAnimationFrame(retry);
            };
            requestAnimationFrame(retry);
        }
    },
});
