# Data Retention Policy

## Chat transcripts
- Product chat transcripts are retained for **90 days** by default.
- Enterprise tenants may request **180-day** retention via contract addendum.
- Users can request deletion; purged records are removed from hot storage within 7 days.

## Embeddings and indexes
- Derived embeddings inherit the retention of their source documents.
- Rebuild indexes after any bulk purge so deleted content cannot be retrieved.

## Audit and security logs
- Audit logs: **365 days** (see Admin > Security > Audit).
- Security IR artifacts: retain per Security policy (minimum 2 years).

## Training use
Customer content is **not** used to train foundation models unless an explicit opt-in is signed.
