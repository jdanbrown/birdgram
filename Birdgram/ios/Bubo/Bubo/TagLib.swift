import Foundation

public enum TagLib {

  // TODO Generalize for Int tags (currently unused, so I haven't bothered)

  public static func readTitle   (_ path: String) throws -> String? { return try _read(path, taglib_tag_title) }
  public static func readArtist  (_ path: String) throws -> String? { return try _read(path, taglib_tag_artist) }
  public static func readAlbum   (_ path: String) throws -> String? { return try _read(path, taglib_tag_album) }
  public static func readComment (_ path: String) throws -> String? { return try _read(path, taglib_tag_comment) }
  public static func readGenre   (_ path: String) throws -> String? { return try _read(path, taglib_tag_genre) }
  // public static func readYear    (_ path: String) throws -> Int?    { return try _read(path, taglib_tag_year) }
  // public static func readTrack   (_ path: String) throws -> Int?    { return try _read(path, taglib_tag_track) }

  public static func writeTitle   (_ path: String, _ value: String) throws { return try _write(path, value, taglib_tag_set_title) }
  public static func writeArtist  (_ path: String, _ value: String) throws { return try _write(path, value, taglib_tag_set_artist) }
  public static func writeAlbum   (_ path: String, _ value: String) throws { return try _write(path, value, taglib_tag_set_album) }
  public static func writeComment (_ path: String, _ value: String) throws { return try _write(path, value, taglib_tag_set_comment) }
  public static func writeGenre   (_ path: String, _ value: String) throws { return try _write(path, value, taglib_tag_set_genre) }
  // public static func writeYear    (_ path: String, _ value: Int)    throws { return try _write(path, value, taglib_tag_set_year) }
  // public static func writeTrack   (_ path: String, _ value: Int)    throws { return try _write(path, value, taglib_tag_set_track) }

  public static func _read(
    _ path: String,
    _ read: (UnsafePointer<TagLib_Tag>?) -> UnsafeMutablePointer<Int8>?
  ) throws -> String? {
    let file = try _file(path)
    defer { taglib_file_free(file) }
    let tag = try _tag(file)
    guard let bytes = read(tag) else {
      throw AppError("Failed to read[\(read)]: \(path)") // FIXME if the read functions can legitimately return nil
    }
    defer { taglib_free(bytes) }
    let value = String(utf8String: bytes)
    return value
  }

  public static func _write(
    _ path: String,
    _ value: String,
    _ write: (UnsafeMutablePointer<TagLib_Tag>?, UnsafePointer<Int8>?) -> Void
  ) throws {
    let file = try _file(path)
    defer { taglib_file_free(file) }
    let tag = try _tag(file)
    value.withCString { value in
      write(tag, value)
    }
    if taglib_file_save(file) == 0 {
      throw AppError(String(format: "Failed to taglib_file_save: path[%@], value[%@]", path, value.slice(to: 1000)))
    }
  }

  public static func _file(_ path: String) throws -> UnsafeMutablePointer<TagLib_File> {
    guard
      let file: UnsafeMutablePointer<TagLib_File> = taglib_file_new(path),
      taglib_file_is_valid(file) != 0
    else {
      throw AppError("File missing or invalid: \(path)")
    }
    return file
  }

  public static func _tag(_ file: UnsafeMutablePointer<TagLib_File>) throws -> UnsafeMutablePointer<TagLib_Tag> {
    guard let tag: UnsafeMutablePointer<TagLib_Tag> = taglib_file_tag(file) else {
      throw AppError("Failed to taglib_file_tag: \(file)")
    }
    return tag
  }
}
