// Neo4j schema initialization for Sonality
// This file is auto-generated from sonality/schema.py - do not edit directly.

CREATE CONSTRAINT episode_uid IF NOT EXISTS FOR (e:Episode) REQUIRE e.uid IS UNIQUE;
CREATE CONSTRAINT derivative_uid IF NOT EXISTS FOR (d:Derivative) REQUIRE d.uid IS UNIQUE;
CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE;
CREATE CONSTRAINT summary_uid IF NOT EXISTS FOR (s:Summary) REQUIRE s.uid IS UNIQUE;
CREATE CONSTRAINT belief_topic IF NOT EXISTS FOR (b:Belief) REQUIRE b.topic IS UNIQUE;
CREATE INDEX episode_created_at IF NOT EXISTS FOR (e:Episode) ON (e.created_at);
CREATE INDEX episode_segment IF NOT EXISTS FOR (e:Episode) ON (e.segment_id);
CREATE INDEX derivative_episode IF NOT EXISTS FOR (d:Derivative) ON (d.source_episode_uid);
