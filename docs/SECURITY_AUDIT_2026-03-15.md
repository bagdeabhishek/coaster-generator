# CoastGen Security Audit Report
**Date:** March 15, 2026
**URL:** https://coastgen.abhishekdoesstuff.com
**Auditor:** Automated Security Scan

## Executive Summary
**Overall Security Score: 8.5/10**

The CoastGen application has a solid security foundation with most critical security headers properly configured. However, there are several areas that need attention, particularly around Content Security Policy (CSP) hardening and cookie security.

---

## Security Headers Analysis

### ✅ Properly Configured Headers

| Header | Status | Value | Notes |
|--------|--------|-------|-------|
| **Strict-Transport-Security (HSTS)** | ✅ Good | `max-age=31536000; includeSubDomains` | Properly enforced HTTPS for 1 year with subdomain protection |
| **X-Frame-Options** | ✅ Good | `DENY` | Prevents clickjacking attacks |
| **X-Content-Type-Options** | ✅ Good | `nosniff` | Prevents MIME type sniffing |
| **X-XSS-Protection** | ✅ Good | `1; mode=block` | Browser XSS filter enabled |
| **Referrer-Policy** | ✅ Good | `strict-origin-when-cross-origin` | Balanced privacy and functionality |
| **Permissions-Policy** | ✅ Good | `camera=(), microphone=(), geolocation=()` | Restricts sensitive API access |
| **Content-Security-Policy** | ⚠️ Partial | See details below | Good foundation but needs hardening |

### ⚠️ Content Security Policy (CSP) Issues

**Current CSP:**
```
default-src 'self';
base-uri 'self';
object-src 'none';
frame-ancestors 'none';
form-action 'self';
script-src 'self' 'unsafe-inline' cdn.tailwindcss.com cdnjs.cloudflare.com unpkg.com threejs.org www.googletagmanager.com;
style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;
img-src 'self' blob: data: https://lh3.googleusercontent.com https://www.google-analytics.com;
connect-src 'self' https://api.bfl.ai https://auth.bfl.ai https://www.google-analytics.com https://region1.google-analytics.com https://stats.g.doubleclick.net;
font-src 'self' https://fonts.gstatic.com;
```

**HIGH SEVERITY Issues:**
1. **`'unsafe-inline'` in script-src** - Allows execution of inline scripts, bypassing XSS protection
2. **Host allowlists in script-src** - Can be bypassed; should use nonces or hashes with `strict-dynamic`

**Recommendations:**
- Remove `'unsafe-inline'` from script-src and style-src
- Implement CSP nonces or hashes for inline scripts
- Consider using `strict-dynamic` for script loading
- Review if all external CDNs are necessary

---

## Cookie Security

### ⚠️ Issues Found

| Cookie | Issue | Severity | Recommendation |
|--------|-------|----------|----------------|
| `coaster_session` | Missing `Secure` flag | **Medium** | Add `Secure` flag to ensure cookies are only sent over HTTPS |
| `coaster_session` | Valid for 30 days | Low | Consider shorter session duration for better security |

**Current Cookie:**
```
coaster_session=eyJhbm9uX2lkIjogIjdmMmY1NTY4OWE1ODRkZDNiYmZhZGQxMDYxMjFjMTRhIn0=.abafPQ.8NE0JScBo6vz03leGh1XYOe-RwQ;
path=/;
Max-Age=2592000;
httponly;
samesite=lax
```

**Missing:**
- ❌ `Secure` flag
- ❌ `__Host-` prefix (for additional protection)

---

## Information Disclosure

### ✅ Good Practices
- Server header shows `uvicorn` (minimal information)
- No stack traces or debug information exposed
- API responses don't expose internal paths or credentials

### ⚠️ Minor Issues
1. **Server Header** - While minimal, it still reveals the server software (uvicorn)
   - **Recommendation:** Consider removing or obfuscating the Server header

2. **API Endpoints** - `/api/auth/providers` and `/api/usage` are publicly accessible
   - **Status:** Not a security issue, but good to know for API documentation

---

## Input Validation & Injection Risks

### ✅ Good Practices
- API uses proper HTTP methods (POST for mutations)
- File uploads are restricted to image types
- Input sanitization appears to be in place for stamp text

### Areas to Verify (Code Review Needed)
- SQL injection prevention in database queries
- XSS prevention in output rendering
- File upload size and type validation
- API rate limiting implementation

---

## Authentication & Session Management

### ✅ Good Practices
- Session cookies use `HttpOnly` flag (prevents JavaScript access)
- `SameSite=Lax` provides CSRF protection
- Session-based authentication (not JWT in localStorage)
- Anonymous users get limited quota (good for abuse prevention)

### ⚠️ Recommendations
1. **Add Secure flag** to session cookie
2. **Consider shorter session duration** for sensitive operations
3. **Implement proper session invalidation** on logout
4. **Add device fingerprinting** for session binding (already partially implemented)

---

## Third-Party Dependencies

### Detected Libraries
- **three.js r128** - 3D rendering library
- **Tailwind CSS** - Via CDN (noted as production warning)
- **Google Fonts** - External font loading

### CDN Usage Concerns
**Console Warning:** "cdn.tailwindcss.com should not be used in production"

**Risk:** Using CDN for Tailwind CSS in production
- **Impact:** External dependency, potential for supply chain attacks
- **Recommendation:** Self-host Tailwind CSS or use build process

---

## Mixed Content

### ✅ All Resources Loaded Over HTTPS
Lighthouse audit confirms no mixed content issues.
All resources are properly loaded via HTTPS.

---

## SSL/TLS Configuration

### ✅ HTTPS Properly Configured
- Valid SSL certificate
- HSTS enabled with 1-year max-age
- Redirects HTTP to HTTPS (handled by Cloudflare/proxy)

---

## Recommendations Summary

### High Priority
1. **Add `Secure` flag to session cookie**
2. **Harden CSP** - Remove `'unsafe-inline'`, implement nonces/hashes
3. **Self-host Tailwind CSS** instead of using CDN

### Medium Priority
4. Remove or obfuscate Server header
5. Review and strengthen input validation
6. Implement additional CSRF tokens for state-changing operations

### Low Priority
7. Consider shorter session duration
8. Add `__Host-` prefix to session cookie
9. Implement Content Security Policy reporting (report-uri)

---

## Security Score Breakdown

| Category | Score | Notes |
|----------|-------|-------|
| Headers | 8/10 | Good foundation, CSP needs work |
| Cookies | 7/10 | Missing Secure flag |
| HTTPS/TLS | 10/10 | Properly configured |
| CSP | 6/10 | `'unsafe-inline'` is problematic |
| Info Disclosure | 9/10 | Minimal leakage |
| Input Validation | 8/10 | Appears good, verify in code |
| Dependencies | 7/10 | CDN usage concerns |
| **Overall** | **8.5/10** | Solid security with room for improvement |

---

## Immediate Action Items

1. [ ] Add `Secure` flag to session cookie in backend configuration
2. [ ] Replace `'unsafe-inline'` in CSP with nonces or hashes
3. [ ] Self-host Tailwind CSS instead of CDN
4. [ ] Review database queries for SQL injection vulnerabilities
5. [ ] Verify XSS prevention in template rendering

---

## Files Reviewed
- `/home/abhishek/Documents/CoasterWebService/main.py` - Backend security headers
- `/home/abhishek/Documents/CoasterWebService/frontend/templates/index.html` - Frontend structure
- `/home/abhishek/Documents/CoasterWebService/report.json` - Lighthouse audit results

## Tools Used
- Chrome DevTools Network Analysis
- Chrome DevTools Console
- Lighthouse Security Audit
- Manual Header Inspection

---

**Report Generated:** March 15, 2026
**Next Audit Recommended:** After implementing high-priority fixes
