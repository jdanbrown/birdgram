-- FIXME(manually_create_indexes): We can't locally rebuild mobile payloads for US/CR because we never synced their
-- payload/*/api/ dirs from remote
--  - (i.e. running notebooks/mobile_build_payload_search_recs locally will barf)
--  - TODO Sync remote data/cache/payloads/*/api/ -> gs -> local
--  - (cf. payloads.py, config.py)

create unique index ix_search_recs__source_id on search_recs (source_id);
create unique index ix_search_recs__species__source_id on search_recs (species, source_id);
create unique index ix_search_recs__species__species_species_group__quality__source_id on search_recs (species, species_species_group, quality, source_id);
create unique index ix_search_recs__species_species_group__species__quality__source_id on search_recs (species_species_group, species, quality, source_id);

-- To query tables/indexes with total size
select name, sum(pgsize) from dbstat group by name;

drop index ix_search_recs__source_id;
drop index ix_search_recs__source_id__species;
drop index ix_search_recs__species;
drop index ix_search_recs__species__source_id;
drop index ix_search_recs__species__species_species_group__quality__source_id;
drop index ix_search_recs__species_species_group__species__quality__source_id;

drop index ix_search_recs_source_id;
drop index ix_search_recs_species_source_id;
