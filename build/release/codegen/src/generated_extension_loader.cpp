#include "duckdb/main/extension/generated_extension_loader.hpp"

namespace duckdb{

//! Looks through the CMake-generated list of extensions that are linked into DuckDB currently to try load <extension>
bool TryLoadLinkedExtension(DuckDB &db, const string &extension) {

  if (extension=="tpch") {
      db.LoadStaticExtension<TpchExtension>();
      return true;
  }
  if (extension=="tpcds") {
      db.LoadStaticExtension<TpcdsExtension>();
      return true;
  }
  if (extension=="core_functions") {
      db.LoadStaticExtension<CoreFunctionsExtension>();
      return true;
  }
  if (extension=="parquet") {
      db.LoadStaticExtension<ParquetExtension>();
      return true;
  }
  if (extension=="jemalloc") {
      db.LoadStaticExtension<JemallocExtension>();
      return true;
  }

    return false;
}

vector<string> LinkedExtensions(){
    vector<string> VEC = {
	"tpch",
	"tpcds",
	"core_functions",
	"parquet",
	"jemalloc"
    };
    return VEC;
}


vector<string> LoadedExtensionTestPaths(){
    vector<string> VEC = {
    };
    return VEC;
}

}
