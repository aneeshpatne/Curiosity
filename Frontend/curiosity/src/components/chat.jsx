import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
const Chat = () => {
  return (
    <div className="w-full h-full flex">
      <h1>Chat</h1>
      <div className="absolute bottom-0 w-full text-white p-4 flex space-x-3">
        <Input type="email" placeholder="Email" />
        <Button type="submit">Subscribe</Button>
      </div>
    </div>
  );
};

export default Chat;
