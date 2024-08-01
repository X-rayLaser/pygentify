class Messenger:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type, callable):
        self.subscribers.setdefault(event_type, []).append(callable)

    def publish(self, event):
        etype = event.etype
        receivers = self.subscribers.get(etype, [])
        for callable in receivers:
            callable(event.data)


class Event:
    etype = "generic"

    def __init__(self, data):
        self.data = data


class TokenArrivedEvent(Event):
    etype = "token_arrived"


class GenerationCompleteEvent(Event):
    etype = "generation_complete"


messenger = Messenger()