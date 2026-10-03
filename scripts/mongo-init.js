// MongoDB initialisation for Keke.
//
// Mounted at /docker-entrypoint-initdb.d/mongo-init.js, which the mongo image
// runs once against the MONGO_INITDB_DATABASE database on first start. This
// file previously did not exist, so the bind mount created a stray directory
// and the container failed to start.
db = db.getSiblingDB('keke');

db.createCollection('documents');
db.createCollection('collaboration_events');

// Keep collaboration events queryable by session and time.
db.collaboration_events.createIndex({ session_id: 1, created_at: -1 });
db.documents.createIndex({ session_id: 1 });
