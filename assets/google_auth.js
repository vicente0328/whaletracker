/* google_auth.js
 * Initializes Google Identity Services (GSI) and bridges the credential
 * to Dash via a hidden trigger button.
 *
 * Flow:
 *  1. GSI renders a Sign-in button inside #g_id_signin
 *  2. User clicks → Google popup → handleGoogleCredential() fires
 *  3. Credential stored in window._googleCredential
 *  4. Hidden button #google-cred-trigger is clicked
 *  5. Dash clientside callback reads window._googleCredential → google-cred-store
 *  6. Server callback exchanges it with Firebase → auth-store
 */

window._googleCredential = null;

function handleGoogleCredential(response) {
    window._googleCredential = response.credential;
    var btn = document.getElementById('google-cred-trigger');
    if (btn) btn.click();
}

function initGoogleSignIn() {
    var clientId = window.GOOGLE_CLIENT_ID;
    if (!clientId || !window.google || !window.google.accounts) return;

    try {
        google.accounts.id.initialize({
            client_id:            clientId,
            callback:             handleGoogleCredential,
            auto_select:          false,
            cancel_on_tap_outside: true,
        });

        var container = document.getElementById('g_id_signin');
        if (container && !container.dataset.rendered) {
            container.dataset.rendered = '1';
            google.accounts.id.renderButton(container, {
                theme:          'filled_black',
                size:           'large',
                text:           'signin_with',
                shape:          'rectangular',
                logo_alignment: 'left',
                width:          320,
            });
        }
    } catch (e) {
        console.warn('GSI init error:', e);
    }
}

/* Re-render button whenever the modal opens (Dash may recreate the DOM node) */
var _gsiObserver = new MutationObserver(function () {
    var el = document.getElementById('g_id_signin');
    if (el && !el.dataset.rendered) {
        initGoogleSignIn();
    }
});

document.addEventListener('DOMContentLoaded', function () {
    _gsiObserver.observe(document.body, { childList: true, subtree: true });
    initGoogleSignIn(); // in case DOM already ready
});
