from typing import Dict, Optional, Any, List


class ChaserLibrary:
    """
    Keeps track of Chaser() objects and their display names.

    - add a new chaser with optional name (defaults to "Chaser <id>")
    - rename/edit an existing chaser's name
    - retrieve and list chasers

    Note: This class is intentionally light-touch and does not assume the
    constructor signature of your Chaser class. Pass in an already-constructed
    Chaser instance if you want the library to manage it. If the object has a
    'name' attribute, it will be kept in sync with the stored name.
    """

    def __init__(self) -> None:
        self._items: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 1

    def add_chaser(self, chaser: Optional[Any] = None, name: Optional[str] = None) -> int:
        """
        Add a new Chaser object to the library.

        Args:
            chaser: An instance of your Chaser class (optional). If not provided,
                    the library will still create an entry that can later be
                    associated with a Chaser instance via set_chaser().
            name: Optional display name. Defaults to "Chaser <id>".

        Returns:
            The integer id assigned to the chaser in the library.
        """
        cid = self._next_id
        self._next_id += 1

        display_name = name or f"Chaser {cid}"
        if chaser is not None and hasattr(chaser, "name"):
            # Keep the object's name in sync with the library display name
            try:
                chaser.name = display_name
            except Exception:
                # Ignore if name is read-only
                pass

        self._items[cid] = {
            "id": cid,
            "name": display_name,
            "obj": chaser,
        }
        return cid

    def set_chaser(self, chaser_id: int, chaser: Any) -> None:
        """
        Associate/replace the Chaser instance for an existing library entry.
        """
        item = self._items.get(chaser_id)
        if not item:
            raise KeyError(f"Chaser id {chaser_id} not found")
        item["obj"] = chaser
        # Sync the name into obj.name if possible
        if hasattr(chaser, "name"):
            try:
                chaser.name = item["name"]
            except Exception:
                pass

    def rename_chaser(self, chaser_id: int, new_name: str) -> None:
        """
        Edit the display name of an existing chaser; if the object has a 'name'
        attribute, keep it in sync.
        """
        item = self._items.get(chaser_id)
        if not item:
            raise KeyError(f"Chaser id {chaser_id} not found")
        item["name"] = new_name
        obj = item.get("obj")
        if obj is not None and hasattr(obj, "name"):
            try:
                obj.name = new_name
            except Exception:
                pass

    def get_chaser(self, chaser_id: int) -> Optional[Any]:
        """
        Retrieve the Chaser object for an id, or None if not set.
        """
        item = self._items.get(chaser_id)
        return item.get("obj") if item else None

    def get_name(self, chaser_id: int) -> Optional[str]:
        """
        Retrieve the display name for an id.
        """
        item = self._items.get(chaser_id)
        return item.get("name") if item else None

    def list(self) -> List[Dict[str, Any]]:
        """
        List library entries with id and name.
        """
        return [{"id": i["id"], "name": i["name"]} for i in self._items.values()]

    def remove(self, chaser_id: int) -> None:
        """Remove a chaser from the library."""
        if chaser_id in self._items:
            del self._items[chaser_id]
