import type {Reference} from './storyboard';
export const kitStory = (id: string) => referenceKit.find(k => k.id === id)?.story;
export const referenceKit: { story: 'desert' | 'office'; id: string; name: string; path: string }[] = [
 {story:'desert',id:'desert-station',name:'Location — desert radio station.jpg',path:'/references/station.jpg'},
 {story:'desert',id:'desert-traveler',name:'Character — the traveler.jpg',path:'/references/traveler.jpg'},
 {story:'desert',id:'desert-receiver',name:'Prop — shortwave receiver.jpg',path:'/references/receiver.jpg'},
 {story:'office',id:'drop-library',name:'Location — the library.jpg',path:'/references/drop-library.jpg'},
 {story:'office',id:'drop-courier',name:'Character — the courier.jpg',path:'/references/drop-courier.jpg'},
 {story:'office',id:'drop-watcher',name:'Character — the watcher.jpg',path:'/references/drop-watcher.jpg'},
];
// Pictures that are not generated yet are skipped, so one missing file never blocks the other stories.
export async function loadReferenceKit():Promise<Reference[]>{const loaded=await Promise.all(referenceKit.map(async (item):Promise<Reference|null>=>{try{const response=await fetch(item.path);if(!response.ok)return null;const blob=await response.blob();if(!blob.type.startsWith('image/'))return null;return {id:item.id,name:item.name,kind:'image',blob};}catch{return null;}}));return loaded.filter((r):r is Reference=>r!==null);}
