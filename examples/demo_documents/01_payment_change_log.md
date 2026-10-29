# Production Change Log - March 12, 2026

**Record ID:** CHG-0312  
**Environment:** Production  
**Owner:** Platform Engineering  
**Status:** Rolled back  
**Document type:** Change history  
**Data classification:** Fictional demonstration data

## 09:12 - Payment gateway configuration deployed

The payments-gateway configuration was updated as part of the morning reliability release.

- `authorization_request_timeout_seconds` changed **from 8 seconds to 2 seconds**.
- The shared gateway is used by `checkout-api` and `renewals-worker`.
- The release was intended to fail slow authorization requests sooner and reduce request tail latency.
- The change was released to production at **09:12**.

## 09:23 - Initial alert

An alert reported a sharp increase in authorization timeouts after the release. The platform team linked the alert to incident **INC-0312** for investigation. Timing alone did not establish root cause.

## 10:06 - Rollback

The team restored `authorization_request_timeout_seconds` to **8 seconds**. No other gateway configuration changes were deployed during the rollback window.

## Notes

This change log describes what was deployed and when. See the incident report for observations and the customer-impact notes for user-facing symptoms. Do not infer a precise financial loss from this document.
